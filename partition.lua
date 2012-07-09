local math = require("math") 
local ffi = require("ffi")
local bitop = require("bit")

require("luarocks.loader")
local narray = require("ljarray.narray")
local helpers = require("ljarray.helpers")
local operator = helpers.operator

module(..., package.seeall) -- export all local functions


FeaturePartition = {}              
FeaturePartition.__index = FeaturePartition

FeaturePartition.create = function(X)
-- construct argsort indices of the feature matrix
-- and build initial partition 0
  local fp = {}
  setmetatable(fp, FeaturePartition)

  local X = X:copy("f") -- copy to fortran order for unstridedness
  -- construct argsort indices
  local X_argsort = narray.create(X.shape, narray.int32, "f")
  for f = 0, X.shape[1]-1 do                              
    local line = X:bind(1,f)
    local line_argsort = line:argsort()[1]
    local X_argsort_line = X_argsort:bind(1,f)
    X_argsort_line:assign(line_argsort)
  end
  -- store features and argsort indices
  fp.X = X
  fp.X_argsort = X_argsort
  fp.samples = narray.create({X.shape[0]}, narray.int32) -- stores the assignment of samples to patititons
  fp.start = {}  -- start offset of partition alogn 0 axis of X, X_argsort
  fp.stop = {}   -- end of partition  along axis 0 of X, X_argsort
  fp.size = {}  -- size of partition
  -- construct inital full partition
  fp.samples:assign(0)
  fp.start[0] = 0
  fp.stop[0] = X.shape[0]
  fp.size[0] = fp.stop[0] - fp.start[0]
  fp.n_partitions = 1
  fp.children = {}

  fp.x_argsort_temp = narray.create({X_argsort.shape[0]}, fp.X_argsort.dtype) -- small scratch pad, allocate once
  return fp -- return the initial partition number
end

FeaturePartition.split = function(self, part, feature, split_pos )
-- split the given parition part into left and right part
-- returns:
--    left_number, right_number
--
  assert(self.children[part] == nil, "FeaturePartition.split: already splitted")

   -- register the new left partitions
  local left_part = self.n_partitions
  self.start[left_part] = self.start[part]
  self.stop[left_part] = self.start[part] + split_pos + 1
  self.size[left_part] = self.stop[left_part] - self.start[left_part]
  self.n_partitions = self.n_partitions + 1

  -- register the new right partitions
  local right_part = self.n_partitions
  self.start[right_part] = self.stop[left_part]
  self.stop[right_part] = self.stop[part]
  self.size[right_part] = self.stop[right_part] - self.start[right_part]
  self.n_partitions = self.n_partitions + 1

  -- register new children
  self.children[part] = {left_part, right_part}

  assert(self.size[left_part] + self.size[right_part] == self.size[part])

  -- store the new sample to partition assignment
  for i = self.start[left_part], self.stop[left_part] -1 do
    self.samples:set(self.X_argsort:get(i, feature), left_part)
  end
  for i = self.start[right_part], self.stop[right_part] - 1 do
    self.samples:set(self.X_argsort:get(i, feature), right_part)
  end

  -- move the features to their new positions
  --
  
  for j = 0, self.X_argsort.shape[1]-1 do
    -- first, copy argsort to scratch pad
    for i = self.start[part], self.stop[part]-1 do
      self.x_argsort_temp:set(i, self.X_argsort:get(i,j))
    end                                                    
    local counter_left = self.start[left_part]
    local counter_right = self.start[right_part]
    -- copy argsort indices to left and right partition
    for i = self.start[part], self.stop[part]-1 do
      local sample_number =  self.x_argsort_temp:get(i)
      local sample = self.samples:get(sample_number) 

      if sample == left_part then
        self.X_argsort:set(counter_left,j, sample_number)
        counter_left = counter_left + 1
      elseif sample == right_part then
        self.X_argsort:set(counter_right,j, sample_number)
        counter_right = counter_right + 1
      end
    end
    assert(counter_left == self.stop[left_part])
    assert(counter_right == self.stop[right_part])
  end
  return left_part, right_part
end





