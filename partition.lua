-- extend package.path with path of this .lua file:local filepath = debug.getinfo(1).source:match("@(.*)$") 
local filepath = debug.getinfo(1).source:match("@(.*)$") 
local dir = string.gsub(filepath, '/[^/]+$', '') .. "/"
package.path = dir .. "/?.lua;" .. package.path

local math = require("math") 
local ffi = require("ffi")
local bitop = require("bit")
local jit = require("jit")

require("luarocks.loader")
local array = require("ljarray.array")
local helpers = require("ljarray.helpers")
local operator = helpers.operator

local Partition = {}
Partition.__index = Partition

ffi.cdef([[
  typedef struct {
    int start;
    int stop;
    int parent;
    char left;
    int left_child;
    int right_child;
  } partition_info;
]])

local Partition_info = ffi.typeof("partition_info")
local Partition_info_VLA = ffi.typeof("partition_info[?]")

Partition.create = function(X, y, mask)
  local pt = {}
  setmetatable(pt, Partition)

  -- count number of samples
  local mask_count = 0
  for i = 0, mask.shape[0] -1 do
    if mask.data[i] == 1 then
      mask_count = mask_count + 1
    end
  end

  -- build dense fortran order feature matrix
  pt.X = array.create({mask_count, X.shape[1]}, X.dtype)
  pt.y = array.create({mask_count}, y.dtype)
  local pos = 0
  for i = 0, X.shape[0] do
    if mask.data[i] == 1 then
      pt.y.data[pos] = y.data[i]
      for f = 0, X.shape[1]-1 do
        pt.X:set(pos,f, X:get(i, f))
      end
      pos = pos + 1
    end
  end
  X = pt.X




  pt.partitions = array.create({2*X.shape[0]}, Partition_info_VLA)
  -- create indices array that holds the samples 
  -- of the partitions in dense ranges
  pt.samples = array.arange(0,X.shape[0],array.int32)
  pt.samples_copy = array.create({pt.samples.shape[0]}, array.int32)
  -- preallocate a scratchpad for feature sorting
  pt.values = array.create({X.shape[0]}, X.dtype)

  -- initialize with 1 full partition
  pt.n_partitions = 1
  local root = pt.partitions.data[0]
  root.start = 0
  root.stop = X.shape[0]
  root.left_child = -1
  root.right_child = -1
  root.parent = -1


  pt.root = 0
  return pt
end

-- returns start and stop of the given partition
-- in the samples array.
--
-- @param parition the partition number
-- @return start,stop the range in the samples vector
Partition.range = function(self, partition)
  local start = self.partitions.data[partition].start
  local stop = self.partitions.data[partition].stop
  return start, stop
end

-- returns the size of a partition
--
-- @param parition the partition number
-- @return size of the partition
Partition.size = function(self, partition)
  local p = self.partitions.data[partition] 
  return p.stop - p.start
end


-- splits a partition at a position in two parts
-- @rturns id_left, id_right
Partition.split = function(self, partition, feature, position)
  -- sort partition samples, and really update 
  self:sort(partition,feature, true)
          
  local id_left = self.n_partitions
  assert(self.n_partitions+2 < self.partitions.shape[0])
  local id_right = self.n_partitions + 1

  self.n_partitions = self.n_partitions + 2

  local t_stop = self.partitions.data[partition].stop
  local t_start = self.partitions.data[partition].start
  assert(t_stop-t_start>position)

  local p = self.partitions.data[partition]
  p.left_child = id_left
  p.right_child = id_right

  local p = self.partitions.data[id_left]
  p.start = t_start
  p.stop = t_start + position + 1
  p.parent = partition
  p.left = 1
  p.left_child = -1
  p.right_child = -1

  p = self.partitions.data[id_right]
  p.start = t_start + position + 1
  p.stop = t_stop
  p.parent = partition
  p.left = 0
  p.left_child = -1
  p.right_child = -1

  return id_left, id_right
end

-- sorts a partition according to a feature
Partition.sort = function(self, partition_nr, feature, in_place)
  local partition =  self.partitions.data[partition_nr]

  assert(partition.start >= 0, partition.start)
  assert(partition.stop <= self.samples.shape[0], partition.stop)

  -- copy feature values to scratchpad
  for i = partition.start,  partition.stop-1  do
    local sample = self.samples.data[i]
    self.values.data[i] = self.X:get(sample, feature)
  end
  
  local argsort
  if in_place == true then
    -- argsort samples via scratchpad values
    argsort = self.values:argsort(0,nil,partition.start, partition.stop, self.samples)
    assert(argsort[1] == self.samples)
  else
    for i = partition.start,  partition.stop-1  do
      self.samples_copy.data[i] = self.samples.data[i]
    end
    argsort = self.values:argsort(0,nil,partition.start, partition.stop, self.samples_copy)
  end

  return argsort[1]
end


return Partition
