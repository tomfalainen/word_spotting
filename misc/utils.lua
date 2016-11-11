local cjson = require 'cjson'
local npy4th = require 'npy4th'

utils = {}

function utils.file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

function utils.read_json(path)
  local file = io.open(path, 'r')
  local text = file:read()
  file:close()
  local info = cjson.decode(text)
  return info
end

function utils.write_json(path, j)
  -- API reference http://www.kyne.com.au/~mark/software/lua-cjson-manual.html#encode
  cjson.encode_sparse_array(true, 2, 10)
  local text = cjson.encode(j)
  local file = io.open(path, 'w')
  file:write(text)
  file:close()
end

-- get the stats
function utils.get_stats(images)
   local tmp = images:type('torch.FloatTensor')
   local mi = tmp:mean()
   local sigma = tmp:std()
   local stats = {}
   stats.mi = mi
   stats.sigma = sigma
   return stats
end

-- following functions taken from Elad Hoffer's
-- TripletNet https://github.com/eladhoffer
function utils.ArrangeByLabel(data)
   local numClasses = data.labels:max()
   local Ordered = {}
   for i=1,data.labels:size(1) do
      -- print(i)
      if Ordered[data.labels[i]] == nil then
        Ordered[data.labels[i]] = {}
      end
      table.insert(Ordered[data.labels[i]], i)
   end
   return Ordered
end


function utils.generate_pairs(Ordered, num_pairs)
   local pairs = torch.IntTensor(num_pairs, 3)
   local used = {}
   local nClasses = #Ordered
   -- for key,value in pairs(Ordered) do nClasses = nClasses + 1 end --Have no idea why this is needed and #Ordered doesn't work
   for i=1, num_pairs do
      local c1 = math.random(nClasses)
      while Ordered[c1] == nil or #Ordered[c1] < 2 or used[c1] ~= nil do
         c1 = math.random(nClasses)
      end
      used[c1] = c1

      local c2 = math.random(nClasses)
      while c2 == c1 or Ordered[c2] == nil or #Ordered[c2] < 1 do
        c2 = math.random(nClasses)
      end

      local n1 = math.random(#Ordered[c1])
      local n2 = math.random(#Ordered[c2])

      -- Should pair be from same class or not 2:1 ratio
      lbl = math.random(0,2)
      if ((lbl==0) or (lbl==1)) then
        pairs[i][1] = Ordered[c1][n1]
        pairs[i][2] = Ordered[c2][n2]
        pairs[i][3] = -1
      else
        pairs[i][1] = Ordered[c1][n1]
        pairs[i][2] = Ordered[c1][n1]
        pairs[i][3] = 1
      end
   end

   return pairs
end

function utils.generate_triplets(Ordered, num_triplets)
  local list = torch.IntTensor(num_triplets, 3)
  local used = {}

  local nClasses = #Ordered
  -- for key,value in pairs(Ordered) do nClasses = nClasses + 1 end --Have no idea why this is needed and #Ordered doesn't work
  for i=1, num_triplets do
    local c1 = math.random(nClasses)
    while Ordered[c1] == nil or #Ordered[c1] < 2 or used[c1] ~= nil do
       c1 = math.random(nClasses)
    end
    used[c1] = c1

    local c2 = math.random(nClasses)
    while c2 == c1 or Ordered[c2] == nil or #Ordered[c2] < 1 do
      c2 = math.random(nClasses)
    end
    local n1 = math.random(#Ordered[c1])
    local n2 = math.random(#Ordered[c2])
    local n3 = math.random(#Ordered[c1])

    while n3 == n1 do
      n3 = math.random(#Ordered[c1])
    end

    list[i][1] = Ordered[c1][n1]
    list[i][2] = Ordered[c2][n2]
    list[i][3] = Ordered[c1][n3]
  end

  return list
end

function utils.getopt(opt, key, default_value)
  if default_value == nil and (opt == nil or opt[key] == nil) then
    error('error: required key ' .. key .. ' was not provided in an opt.')
  end
  if opt == nil then return default_value end
  local v = opt[key]
  if v == nil then v = default_value end
  return v
end

--Horrible solution, but can't be bothered to re-implement mAP evaluation in torch.
function utils.map_eval(dataset, id, descrs, upper_bound, fold)
  upper_bound = upper_bound or -1
  npy4th.savenpy('tmp/' .. dataset .. '_' .. id..'_descriptors.npy', descrs)
  if dataset == 'washington' then
    os.execute('ipython misc/map_eval.py ' .. fold .. ' ' .. dataset .. '_' .. id .. ' ' .. upper_bound)
  end

  local result_struct = utils.read_json('tmp/' .. dataset .. '_' .. id .. '_ws_results.json') 
  return result_struct
end

return utils


