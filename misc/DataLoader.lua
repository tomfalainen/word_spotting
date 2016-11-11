require 'hdf5'
local npy4th = require 'npy4th'
local utils = require 'misc.utils'
local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)
  
  -- load the json file which contains additional information about the dataset
  print('DataLoader loading json file: ', opt.root_dir .. opt.dataset .. '_preprocessed.json')
  self.json = utils.read_json(opt.root_dir .. opt.dataset .. '_preprocessed.json')
  
  self.data = {}
  self.data.splits = {}
  for i=1, #self.json.data do
    self.data.splits[i] = self.json.data[i].split  
  end

  -- open the hdf5 file
  print('DataLoader loading h5 file: ', self.json.image_db .. '.h5')
  self.h5_file = hdf5.open(self.json.image_db .. '.h5', 'r')

  -- Copy some variables from opt
  self.root_dir = opt.root_dir

  if opt.mode == 'train' then
    self.pieces = self.json.pieces[1]
    self.current_piece = 1

    self.data.labels = self.h5_file:read('/train_labels_1'):all()
    self.data.images = self.h5_file:read('/train_images_1'):all()

    -- normalize the data by mean subtraction
    self.stats = utils.get_stats(self.data.images)
    -- torch.save('stats.'.. opt.dataset..'.t7', stats)
    npy4th.savenpy(self.json.image_db .. '.mean.npy', torch.Tensor({self.stats.mi}))
    self.data.images:add(-self.stats.mi)

    -- extract image size from dataset
    local images_size = #self.data.images
    print(images_size)
    assert(#images_size == 4, '/train_images should be a 4D tensor')
    self.num_images = images_size[1]
    self.num_channels = images_size[2]
    self.image_height = images_size[3]
    self.image_width = images_size[4]
    print(string.format('read %d images of size %dx%dx%d', self.num_images, 
              self.num_channels, self.image_height, self.image_width))

    -- Some utilities for random triplet sampling
    self.ordered_data = utils.ArrangeByLabel(self.data)
     
    -- Initialize iterators, used for get_batch
    self.iterators = {}
    self.iterators['train'] = 1

  else
    self.data.labels = self.h5_file:read('/labels'):all()
    self.data.images = self.h5_file:read('/images'):all()
    local stats = utils.get_stats(self.data.images) -- slightly different mean from the training dataset, but shouldn't matter.
    self.data.images:add(-stats.mi)

    -- extract image size from dataset
    local images_size = #self.data.images
    print(images_size)
    assert(#images_size == 4, '/images should be a 4D tensor')
    self.num_images = images_size[1]
    self.num_channels = images_size[2]
    self.image_height = images_size[3]
    self.image_width = images_size[4]
    print(string.format('read %d images of size %dx%dx%d', self.num_images, 
              self.num_channels, self.image_height, self.image_width))
  end

end

function DataLoader:get_data()
  return self.data
end

function DataLoader:resetIterator(split)
  self.iterators[split] = 1
end

function DataLoader:switch_data_piece(piece)
  print("switching to data piece " .. piece)
  -- Load next data piece
  self.h5_file = hdf5.open(self.json.image_db .. '.h5', 'r')
  self.data.labels = self.h5_file:read('/train_labels_' .. piece):all()
  self.data.images = self.h5_file:read('/train_images_' .. piece):all()
  self.h5_file:close()

  -- Subtract training mean and update some variables
  self.data.images:add(-self.stats.mi)
  local images_size = #self.data.images
  self.num_images = images_size[1]
  self.current_piece = piece
  self.ordered_data = utils.ArrangeByLabel(self.data)
end

function DataLoader:getBatch(opt)
  local batch_size = utils.getopt(opt, 'batch_size') -- how many images get returned at one time (to go through CNN)

  -- Randomly sample triplets
  local t = utils.generate_triplets(self.ordered_data, batch_size)

  local x=torch.zeros(batch_size,1,60,160):cuda()
  local y=torch.zeros(batch_size,1,60,160):cuda()
  local z=torch.zeros(batch_size,1,60,160):cuda()

  local split = 'train'
  local k = self.iterators[split]
  for i=1,batch_size do 
    if self.pieces > 1 and k > self.num_images then 
      self:switch_data_piece((self.current_piece % self.pieces) + 1) 
      k = 1
    end
    x[i] = self.data.images[t[i][1]]
    y[i] = self.data.images[t[i][2]]
    z[i] = self.data.images[t[i][3]]
    k = k + 1
  end
  self.iterators[split] = k

  local data = {x, y, z}
  return data
end
