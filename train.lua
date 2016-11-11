require 'torch'
require 'nn'
require 'math'
require 'optim'

-- local imports
local utils = require 'misc.utils'
require 'misc.DataLoader'
require 'misc.DistanceRatioCriterion'
local models = require 'misc.models'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an Word Spotting model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-dataset','','path to the h5file containing the preprocessed dataset')
cmd:option('-root_dir','data/washington/','path to data root directory')
cmd:option('-weights', 'pretrained/presnet_34_cvl_iter_185000.t7', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')

-- Model settings
cmd:option('-descriptor_size', 512,'the dimensionality of the descriptors (when applicable).')
cmd:option('-fold', 1,'Which fold of washington data to run')
cmd:option('-val_upper_bound',1000,'number of examples to use for MAP evaluation')
cmd:option('-val_dataset', 'washington','Dataset to evaluate on')

-- Optimization:
cmd:option('-max_iters', 50000, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-batch_size',128,'what is the batch size in number of images per batch?')
cmd:option('-learning_rate',1e-2,'learning rate')
cmd:option('-learning_rate_decay_every', 20000, 'every how many iterations thereafter to drop LR by learning_rate_decay_factor')
cmd:option('-learning_rate_decay_factor', 0.1, 'how much to decay the learning rate with')

-- Evaluation/Checkpointing
cmd:option('-save_checkpoint_every', 10000, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', 'checkpoints', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-losses_log_every', 100, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
cmd:option('-print_train_loss_every', 500, 'How often do we print losses (0 = disable)')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', '', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')

cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then 
    require 'cudnn' 
    cudnn.benchmark = true
  end
  cutorch.manualSeed(opt.seed)
  math.randomseed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

-------------------------------------------------------------------------------
-- Initialize the networks
-------------------------------------------------------------------------------
local protos = {}
if string.len(opt.weights) > 0 then
  -- Load checkpoint
  print('initializing weights from ' .. opt.weights)
  cp = torch.load(opt.weights)
  protos = cp.protos
else -- create protos from scratch

  -- intialize model
  protos.cnn = models.load_presnet(opt.descriptor_size)
  protos.mlp = models.load_triplet_net(protos.cnn, opt)
  cp = {}
end

-- Load criterion 
protos.crit = nn.DistanceRatioCriterion()

-- flatten and prepare all model parameters to a single vector. 
local params, grad_params = protos.mlp:getParameters()
print('total number of parameters in model: ', params:nElement())
assert(params:nElement() == grad_params:nElement())

collectgarbage() -- "yeah, sure why not"

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
opt.mode = 'train'
print("loading training database")
loader = DataLoader(opt)

-- load validation dataloader
print("Loading validation database")
val_opt = {}
val_opt.root_dir = opt.root_dir
val_opt.mode  = 'full'
val_opt.dataset = opt.val_dataset

local val_loader = DataLoader(val_opt)

-- ship everything to GPU, maybe
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end

-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function map_eval()
  protos.mlp:evaluate()
  local data = val_loader:get_data()
  local N = data.images:size(1)
  local cnn = protos.mlp.modules[1].modules[1]

  descrs = torch.CudaTensor(N, opt.descriptor_size)
  descrs_split = descrs:split(opt.batch_size)
  for i,v in ipairs(data.images:split(opt.batch_size)) do
    descrs_split[i]:copy(cnn:forward(v:cuda()))
  end

  result = utils.map_eval(val_opt.dataset, opt.id, descrs, opt.val_upper_bound, opt.fold)
  return result
end

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------

optimState = {
  learningRate = opt.learning_rate,
  weightDecay = 1e-4,
  momentum = 0.9,
  dampening = 0,
  nesterov = 1,
}

local iter = opt.iter or 0
local inputs
local targets
local loss
local loss_history = cp.loss_history or {}
local map_history = cp.map_history or {}
local running_average_loss = 0
local best_score
local outputs
local cnn_outputs
local learning_rate = opt.learning_rate

map_result = map_eval()
print('validation desrc MaP: ' .. map_result.MaP_qbe)
map_history[iter] = map_result.MaP_qbe

print('starting main loop')
while true do
  local feval = function(f)
    if f ~= params then params:copy(f) end
    grad_params:zero()

    protos.mlp:training()
    inputs = loader:getBatch{batch_size = opt.batch_size}

    outputs = protos.mlp:forward(inputs)
    loss = protos.crit:forward(outputs, 1)
    df_do = protos.crit:backward(outputs)
    protos.mlp:backward(inputs, df_do)
    return losses,grad_params
  end

  optimState.learningRate = learning_rate
  optim.sgd(feval, params, optimState)
  losses = {}
  losses.total_loss = loss

  -- Record loss in various ways
  running_average_loss = running_average_loss + losses.total_loss
  
  -- Optional printing and logging of losses
  if iter % opt.losses_log_every == 0 then loss_history[iter] = losses.total_loss end

  if opt.print_train_loss_every > 0 and iter % opt.print_train_loss_every == 0 then
    running_average_loss = running_average_loss / opt.print_train_loss_every
    print(string.format('iter %d: %f, lr=%f', iter, running_average_loss, learning_rate ))
    running_average_loss = 0
  end

  -- save checkpoint once in a while (or on final iteration)
  if (iter > 0 and iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then

    -- do MaP eval
    map_result = map_eval()
    print('validation MaP: ' .. map_result.MaP_qbe)
    map_history[iter] = map_result.MaP_qbe

    local checkpoint_path = path.join(opt.checkpoint_path, opt.id .. '_iter_' .. iter)

    -- write a (thin) json report
    local checkpoint = {}
    checkpoint.opt = opt
    checkpoint.iter = iter
    checkpoint.loss_history = loss_history
    checkpoint.map_history = map_history
    checkpoint.learning_rate = learning_rate

    utils.write_json(checkpoint_path .. '.json', checkpoint)
    print('wrote json checkpoint to ' .. checkpoint_path .. '.json')

    -- write the full model checkpoint as well if we did better than ever
    local current_score = map_result.MaP_qbe
    -- if best_score == nil or current_score > best_score then
      best_score = current_score
      if iter > 0 then -- dont save on very first iteration
        -- include the protos (which have weights) and save to file
        
        protos.mlp:clearState()
        local save_protos = {}
		    save_protos.mlp = protos.mlp
        checkpoint.protos = save_protos
        checkpoint.iter = iter
        torch.save(checkpoint_path .. '.t7', checkpoint)
        print('wrote checkpoint to ' .. checkpoint_path .. '.t7')
      end
    -- end
  end

  if iter % opt.learning_rate_decay_every == 0 and iter > 0 then
    learning_rate = learning_rate * opt.learning_rate_decay_factor -- set the decayed rate
  end

  -- stopping criterions
  iter = iter + 1
  if iter % 10 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
  if opt.max_iters > 0 and iter > opt.max_iters then break end -- stopping criterion

end
