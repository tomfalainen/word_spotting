require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nn'
require 'torch'
require 'misc.DataLoader'
npy4th = require 'npy4th'

opt = {}
opt.dataset = 'washington'
opt.mode = 'full'
opt.root_dir = 'data/washington/'

loader = DataLoader(opt)
data = loader:get_data()
N = data.images:size(1)
folds = 1
embeddings = {'triplet', 'dct3', 'ngram', 'semantic', 'phoc'}

for i = 1, #embeddings do
	embedding = embeddings[i]
	iter = 30000
	for fold = 1, folds do
	 	if embedding == 'triplet' then
			iter = 50000
		end
		
		model = 'checkpoints/washington_' .. embedding .. '_fold_' .. fold .. '_iter_' .. iter .. '.t7'
		
		loaded_checkpoint = torch.load(model)
		protos = loaded_checkpoint.protos
        protos.mlp:cuda()

	 	if embedding == 'triplet' then
	 		cnn = protos.mlp.modules[1].modules[1]
	 	else 
	 		cnn = protos.mlp.modules[1]
	 	end

		print('loaded network ' .. model)
		 
		-- extract the cnn to get the descriptors from a network
		cnn:evaluate()
		ds = #cnn:forward(torch.CudaTensor(1, 1, 60, 160))
        descriptor_size = ds[2]

		BatchSize = 128
		descriptors = torch.CudaTensor(N, descriptor_size)
		descriptor_split = descriptors:split(BatchSize)
		for i,v in ipairs(data.images:split(BatchSize)) do
			descriptor_split[i]:copy(cnn:forward(v:cuda()))
		end

		-- Save descriptors for further processing in python
		npy4th.savenpy('descriptors/washington_' .. embedding .. '_fold_' .. fold .. '_descriptors.npy', descriptors)

	end
end
