require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'
local createModel = require 'misc.presnet'

local models = {}

function models.load_presnet(descriptor_size)
    local opt = {}
    opt.dataset = 'imagenet'
    opt.depth = 34
    opt.descriptor_size = descriptor_size
    local model = createModel(opt)
    return model
end


function models.load_embedding_net(cnn, opt)
    local net = nn.Sequential()

    net:add(nn.Linear(opt.descriptor_size, opt.fc_size))
    net.modules[#net.modules].weight:normal(0, 0.01)
    net.modules[#net.modules].bias:fill(0.1)
    net:add(nn.BatchNormalization(opt.fc_size))
    net:add(nn.Tanh(true))

    net:add(nn.Linear(opt.fc_size, opt.fc_size))
    net.modules[#net.modules].weight:normal(0, 0.01)
    net.modules[#net.modules].bias:fill(0.1)
    net:add(nn.BatchNormalization(opt.fc_size))
    net:add(nn.Tanh(true))

    net:add(nn.Linear(opt.fc_size, opt.fc_size))
    net.modules[#net.modules].weight:normal(0, 0.01)
    net.modules[#net.modules].bias:fill(0.1)
    net:add(nn.BatchNormalization(opt.fc_size))
    net:add(nn.Tanh(true))

    net:add(nn.Linear(opt.fc_size, opt.embedding_size))
    net.modules[#net.modules].weight:normal(0, 0.01)
    net.modules[#net.modules].bias:fill(0.1)
    net:add(nn.Normalize(2))
    local mlp = nn.Sequential()
    mlp:add(cnn)
    mlp:add(net)
    local prl = nn.ParallelTable()
    prl:add(mlp)
    prl:add(nn.Identity())

    return prl
end

function models.load_triplet_net(cnn, opt)
    --  clone the other two networks in the triplet
    cnn2 = cnn:clone('weight', 'bias','gradWeight','gradBias', 'running_mean', 'running_var') -- for batch norm
    cnn3 = cnn:clone('weight', 'bias','gradWeight','gradBias', 'running_mean', 'running_var') -- for batch norm

    -- add them to a parallel table
    prl = nn.ParallelTable()
    prl:add(cnn)
    prl:add(cnn2)
    prl:add(cnn3)
    prl:cuda()

    mlp= nn.Sequential()
    mlp:add(prl)

    -- get feature distances 
    cc = nn.ConcatTable()

    -- feats 1 with 2 
    cnn_left = nn.Sequential()
    cnnpos_dist = nn.ConcatTable()
    cnnpos_dist:add(nn.SelectTable(1))
    cnnpos_dist:add(nn.SelectTable(2))
    cnn_left:add(cnnpos_dist)
    if opt.use_metric_net then
      cnn_left:add(metric1)    
    else
      cnn_left:add(nn.PairwiseDistance(2))
    end
    cnn_left:add(nn.View(opt.batch_size,1))
    cnn_left:cuda()
    cc:add(cnn_left)

    -- feats 2 with 3 
    cnn_left2 = nn.Sequential()
    cnnpos_dist2 = nn.ConcatTable()
    cnnpos_dist2:add(nn.SelectTable(2))
    cnnpos_dist2:add(nn.SelectTable(3))
    cnn_left2:add(cnnpos_dist2)
    if opt.use_metric_net then
      cnn_left2:add(metric1)    
    else
      cnn_left2:add(nn.PairwiseDistance(2))
    end
    cnn_left2:add(nn.View(opt.batch_size,1))
    cnn_left2:cuda()
    cc:add(cnn_left2)

    -- feats 1 with 3 
    cnn_right = nn.Sequential()
    cnnneg_dist = nn.ConcatTable()
    cnnneg_dist:add(nn.SelectTable(1))
    cnnneg_dist:add(nn.SelectTable(3))
    cnn_right:add(cnnneg_dist)
    if opt.use_metric_net then
      cnn_right:add(metric1)    
    else
      cnn_right:add(nn.PairwiseDistance(2))
    end
    cnn_right:add(nn.View(opt.batch_size,1))
    cnn_right:cuda()
    cc:add(cnn_right)
    cc:cuda()

    mlp:add(cc)

    last_layer = nn.ConcatTable()

    -- select min negative distance inside the triplet
    mined_neg = nn.Sequential()
    mining_layer = nn.ConcatTable()
    mining_layer:add(nn.SelectTable(1))
    mining_layer:add(nn.SelectTable(2))
    mined_neg:add(mining_layer)
    mined_neg:add(nn.JoinTable(2))
    mined_neg:add(nn.Min(2))
    mined_neg:add(nn.View(opt.batch_size,1))
    last_layer:add(mined_neg)
    -- add positive distance
    pos_layer = nn.Sequential()
    pos_layer:add(nn.SelectTable(3))
    pos_layer:add(nn.View(opt.batch_size,1))
    last_layer:add(pos_layer)

    mlp:add(last_layer)

    mlp:add(nn.JoinTable(2))
    mlp:cuda()
    return mlp
end

return models