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

FeaturePartition.create = function(X, y, n_classes)
-- construct argsort indices of the feature matrix
-- and build initial partition 0
  local fp = {}
  setmetatable(fp, FeaturePartition)

  local X = X:copy("f") -- copy to fortran order for unstridedness
  local X_argsort = narray.create(X.shape, narray.int32, "f")

  -- store features and argsort indices
  fp.X = X
  fp.f_sorted = helpers.zeros(X.shape[1])
  fp.X_argsort = X_argsort
  fp.y = y

  -- sample_partition stores the assignment of samples to patititons
  fp.sample_partition = narray.create({X.shape[0]}, narray.int32)
  fp.n_samples = X.shape[0]

  
  -- construct inital full partition
  fp.sample_partition:assign(0)
  fp.n_partitions = 1
  fp.n_classes = n_classes
  fp.partition_class_count = narray.create({2*fp.n_samples, fp.n_classes + 1}, narray.int32, "f")
  fp.size = narray.create({2*fp.n_samples}, narray.int32)

  fp:reset()
  return fp -- return the initial partition number
end

FeaturePartition.reset = function(self)
  self.sample_partition:assign(0)
  self.n_partitions = 1
  self.size:assign(0)
  self.size:set(0,self.n_samples)
  self:update(self.sample_partition)
end

FeaturePartition.ensure_sorted = function(self,f)
  if self.f_sorted[f] == 0 then
    print("sorting feature", f)
    local line = self.X:bind(1,f)
    local line_argsort = line:argsort()[1]
    local X_argsort_line = self.X_argsort:bind(1,f)
    X_argsort_line:assign(line_argsort)
    self.f_sorted[f] = 1
  end
end

FeaturePartition.update = function(self, sample_partition) 
  assert(sample_partition.shape[0] == self.n_samples)

  self.size:assign(0)
  self.n_partitions = 0
  self.partition_class_count:assign(0)

  -- initialize sample_class_count
  for i = 0, self.n_samples-1 do
    local class = self.y:get(i)
    local cur_part = sample_partition:get(i)
    local cur_count = self.partition_class_count:get(cur_part,class)
    self.partition_class_count:set(cur_part, class, cur_count +1)

    -- update partition sizes
    local size = self.size:get(cur_part)
    self.size:set(cur_part,  size + 1)

    -- update number of partitions
    if cur_part > self.n_partitions then
      self.n_partitions = cur_part
    end
  end


  self.n_partitions = self.n_partitions + 1
  if self.n_partitions >= 2*self.n_samples then
    print(self.n_partitions, 2*self.n_samples)
  end
  assert(self.n_partitions < 2*self.n_samples)

  self.sample_partition:assign(sample_partition)
end
