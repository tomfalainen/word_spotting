require 'torch'
require 'nn'
require 'math'
-- local imports
utils = require 'misc.utils'
require 'misc.DataLoader'
require 'misc.DataLoader_we'
-- require 'misc.DistanceRatioCriterion'
models = require 'misc.models'
require 'optim'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an Word Spotting model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-dataset','','path to the h5file containing the preprocessed images')
cmd:option('-root_dir','data/washington/','path to data root directory')
cmd:option('-weights', 'checkpoints/washington_triplet_fold_1_iter_50000.t7', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')

cmd:option('-fold', 1,'Which fold of washington data to run')
cmd:option('-val_upper_bound',1000,'number of examples to use for MAP evaluation')
cmd:option('-val_dataset', 'washington','number of examples to use for MAP evaluation')

-- Optimization:
cmd:option('-max_iters', 30000, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-batch_size',128,'what is the batch size in number of images per batch?')
cmd:option('-learning_rate',1e-2,'learning rate')
cmd:option('-learning_rate_decay_every', 10000, 'every how many iterations thereafter to drop LR by learning_rate_decay_factor')
cmd:option('-learning_rate_decay_factor', 0.1, 'how much to decay the learning rate with')
cmd:option('-cosine_margin',0.1,'the margin for the cosine embedding loss')

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
opt = cmd:parse(arg)
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
-- Create the Data Loader instance
-------------------------------------------------------------------------------

print("loading training database")
opt.mode = 'train'
loader = DataLoaderWE(opt)

-- load validation dataloader
print("Loading validation database")
val_opt = {}
val_opt.root_dir = opt.root_dir
val_opt.dataset = opt.val_dataset
val_opt.mode = 'full'
val_opt.upper_bound = opt.val_upper_bound
val_loader = DataLoader(val_opt)

-------------------------------------------------------------------------------
-- Initialize the networks
-------------------------------------------------------------------------------
protos = {}
cp = {}
opt.descriptor_size = 512
opt.fc_size = 4096
opt.embedding_size = loader:get_embedding_size()
if string.len(opt.weights) > 0 then
    -- Load checkpoint
  print('initializing weights from ' .. opt.weights)
  lcp = torch.load(opt.weights)
  protos = lcp.protos
  
  protos.cnn = protos.mlp.modules[1].modules[1]
  protos.mlp = models.load_embedding_net(protos.cnn, opt)
  protos.mlp:cuda()
else
  -- create protos from scratch, intialize model
  protos.cnn = models.load_presnet(opt.descriptor_size)
  protos.mlp = models.load_embedding_net(protos.cnn, opt)
end

-- Load cosine embedding criterion
protos.crit = nn.CosineEmbeddingCriterion(opt.cosine_margin)

-- ship everything to GPU, maybe
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end

-- flatten and prepare all model parameters to a single vector. 
local params, grad_params = protos.mlp:getParameters()
print('total number of parameters in model: ', params:nElement())
assert(params:nElement() == grad_params:nElement())

collectgarbage() -- "yeah, sure why not"
-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function map_eval()
  protos.mlp:evaluate()
  local data = val_loader:get_data()
  local N = data.images:size(1)
  local descrs = torch.CudaTensor(N, loader:get_embedding_size())
  local descrs_split = descrs:split(opt.batch_size)
  for i,v in ipairs(data.images:split(opt.batch_size)) do
    descrs_split[i]:copy(protos.mlp.modules[1]:forward(v:cuda()))
  end

  local result = utils.map_eval(val_opt.dataset, opt.id, descrs, val_opt.upper_bound, opt.fold)
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

-- Define some variables to keep track of things
local iter = opt.iter or 0
local inputs
local embeddings
local targets
local loss
local loss_history = cp.loss_history or {}
local map_qbe_history = cp.map_qbe_history or {}
local map_qbs_history = cp.map_qbs_history or {}
local epoch_count = 0
local running_average_loss = 0
local best_score
local learning_rate = opt.learning_rate
local finetune_cnn = 1

map_result = map_eval()
print('validation MaP qbe: ' .. map_result.MaP_qbe .. ', qbs: ' .. map_result.MaP_qbs)
map_qbe_history[iter] = map_result.MaP_qbe
map_qbs_history[iter] = map_result.MaP_qbs

print('starting main loop')
while true do

  local feval = function(f)
    if f ~= params then params:copy(f) end
    grad_params:zero()

    --Load data
    protos.mlp:training()
    inputs, embeddings, targets = unpack(loader:getBatch{batch_size = opt.batch_size})

    local cnn_outputs = protos.mlp.modules[1].modules[1]:forward(inputs)
    local outputs = protos.mlp.modules[1].modules[2]:forward(cnn_outputs)
    loss = protos.crit:forward({outputs, embeddings}, targets)  
    local df_do, dummy = unpack(protos.crit:backward({outputs, embeddings}, targets))
    local dcnn_outputs = protos.mlp.modules[1].modules[2]:backward(cnn_outputs, df_do)  
    if finetune_cnn > 0 and iter > 500 then 
      dx = protos.mlp.modules[1].modules[1]:backward(inputs, dcnn_outputs)
    end
    return loss, grad_params
  end

  optimState.learningRate = learning_rate
  optim.sgd(feval, params, optimState)

  -- Record loss in various ways
  running_average_loss = running_average_loss + loss
  
  -- Optional printing and logging of losses
  if iter % opt.losses_log_every == 0 then loss_history[iter] = loss end

  if opt.print_train_loss_every > 0 and iter % opt.print_train_loss_every == 0 then
    running_average_loss = running_average_loss / opt.print_train_loss_every
    print(string.format('iter %d: %f, lr=%f', iter, running_average_loss, learning_rate ))
    running_average_loss = 0
  end

  -- save checkpoint once in a while (or on final iteration)
  if (iter > 0 and iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then

    -- do MaP eval
    local map_result = map_eval()

    print('validation MaP qbe: ' .. map_result.MaP_qbe .. ', qbs: ' .. map_result.MaP_qbs)
    map_qbe_history[iter] = map_result.MaP_qbe
    map_qbs_history[iter] = map_result.MaP_qbs

    local checkpoint_path = path.join(opt.checkpoint_path, opt.id .. '_iter_' .. iter)

    -- write a (thin) json report
    local checkpoint = {}
    checkpoint.opt = opt
    checkpoint.iter = iter
    checkpoint.loss_history = loss_history
    checkpoint.map_qbe_history = map_qbe_history
    checkpoint.map_qbs_history = map_qbs_history
    checkpoint.learning_rate = learning_rate

    utils.write_json(checkpoint_path .. '.json', checkpoint)
    print('wrote json checkpoint to ' .. checkpoint_path .. '.json')

    -- write the full model checkpoint as well if we did better than ever
    local current_score = (map_result.MaP_qbe + map_result.MaP_qbs) / 2
    -- if best_score == nil or current_score > best_score then
      best_score = current_score
      if iter > 0 then -- dont save on very first iteration

        protos.mlp:clearState()
          local save_protos = {}
        save_protos.mlp = protos.mlp
        checkpoint.protos = save_protos
        checkpoint.model_type = opt.model_type
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
