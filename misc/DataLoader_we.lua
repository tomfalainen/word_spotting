require 'hdf5'
local utils = require 'misc.utils'
local npy4th = require 'npy4th'
local DataLoaderWE = torch.class('DataLoaderWE')

function DataLoaderWE:__init(opt)
  
  -- load the json file which contains additional information about the dataset
  print('DataLoaderWE loading json file: ', opt.root_dir .. opt.dataset .. '_preprocessed.json')
  self.json = utils.read_json(opt.root_dir .. opt.dataset .. '_preprocessed.json')
  
  self.data = {}
  self.data.splits = {}
  for i=1, #self.json.data do
    self.data.splits[i] = self.json.data[i].split  
  end

  -- open the hdf5 file
  self.image_h5_file = hdf5.open(self.json.image_db .. '.h5', 'r')
  self.embedding_h5_file = hdf5.open(self.json.embedding_db .. '.h5', 'r')
  self.root_dir = opt.root_dir

  if opt.mode == 'train' then
    self.pieces = self.json.pieces[1]
    self.current_piece = 1
    
    --Load the first piece, or only piece if there is only one
    self.data.embeddings = self.embedding_h5_file:read('/train_embeddings_1'):all() 
    self.data.labels = self.image_h5_file:read('/train_labels_1'):all()
    self.data.images = self.image_h5_file:read('/train_images_1'):all()

    -- normalize the data by mean subtraction
    self.stats = utils.get_stats(self.data.images)
    -- torch.save('stats.'.. opt.dataset..'.t7', stats)
    npy4th.savenpy(self.json.image_db .. '.mean.npy', torch.Tensor({self.stats.mi}))
    self.data.images:add(-self.stats.mi)

    -- extract image size from dataset
    local train_images_size = #self.data.images
    assert(#train_images_size == 4, '/train_images should be a 4D tensor')
    self.num_images = train_images_size[1]
    self.num_channels = train_images_size[2]
    self.image_height = train_images_size[3]
    self.image_width = train_images_size[4]
    print(string.format('read %d images of size %dx%dx%d', self.num_images, self.num_channels, self.image_height, self.image_width))

    self.embedding_size = (#self.data.embeddings)[2]
    self.ordered_data = utils.ArrangeByLabel(self.data)
     
    -- Initialize iterators, used for get_batch
    self.iterators = {}
    self.iterators['train'] = 1

  end
  self.image_h5_file:close()
  self.embedding_h5_file:close()
end

function DataLoaderWE:get_embedding_size()
  return self.embedding_size
end

function DataLoaderWE:get_data()
  return self.data
end

function DataLoaderWE:resetIterator(split)
  self.iterators[split] = 1
end

function DataLoaderWE:switch_data_piece(piece)
  print("switching to data piece " .. piece)
  -- Load next data piece
  self.image_h5_file = hdf5.open(self.json.image_db .. '.h5', 'r')
  self.embedding_h5_file = hdf5.open(self.json.embedding_db .. '.h5', 'r')
  self.data.embeddings = self.embedding_h5_file:read('/train_embeddings_' .. piece):all() 
  self.data.labels = self.image_h5_file:read('/train_labels_' .. piece):all()
  self.data.images = self.image_h5_file:read('/train_images_' .. piece):all()
  self.image_h5_file:close()
  self.embedding_h5_file:close()

  -- Subtract training mean and update some variables
  self.data.images:add(-self.stats.mi)
  local train_images_size = #self.data.images
  self.num_images = train_images_size[1]
  self.current_piece = piece
  self.ordered_data = utils.ArrangeByLabel(self.data)
end

--[[
    Returns a batch of data:
  - x (N, 1, 60, 160) containing the images
  - y (N, M) containing the embeddings of dimension M
  - z (N) Labels for whether or not pairs belong to same class, used for cosine embedding loss
--]]
function DataLoaderWE:getBatch(opt)
  local batch_size = utils.getopt(opt, 'batch_size') -- how many images get returned at one time 

  -- Sample random pairs
  self.data.pairs = utils.generate_pairs(self.ordered_data, batch_size)

  local x=torch.zeros(batch_size,1,60,160):cuda() -- Images, hard coded size
  local y=torch.zeros(batch_size, self.embedding_size):cuda()    -- Word embeddings
  local z=torch.zeros(batch_size):cuda()          -- Labels, in {-1, 1}

  local split = 'train'
  local k = self.iterators[split]
  for i=1,batch_size do 
    if self.pieces > 1 and k > self.num_images then 
      self:switch_data_piece((self.current_piece % self.pieces) + 1) 
      k = 1
    end
    local t = self.data.pairs
    x[i] = self.data.images[t[i][1]]
    y[i] = self.data.embeddings[t[i][2]]
    z[i] = t[i][3]
    k = k + 1
  end
  self.iterators[split] = k

  local data = {x, y, z}
  return data

end
