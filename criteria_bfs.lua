local math = require("math") 
local ffi = require("ffi")
local bitop = require("bit")

require("luarocks.loader")
local narray = require("ljarray.narray")
local helpers = require("ljarray.helpers")
local operator = helpers.operator

module(..., package.seeall) -- export all local functions

ffi.cdef[[
  typedef struct { int size; int n_left; int n_right; float H_left; float H_right; int position; float best_gini; int best_pos; float best_x; int clean;} gini_info;

  typedef struct {int sample; int partition; int class;} sample_info;
]]

local Gini_info = ffi.typeof("gini_info[?]")
local Sample_info = ffi.typeof("sample_info[?]")

Gini = {}
Gini.__index = Gini

Gini.create = function(feature_partition)
  local gini = {}
  setmetatable(gini, Gini)
  gini.gini_info_t = Gini_info
  gini.feature_partition = feature_partition
  gini.n_classes = feature_partition.n_classes
  gini.y = feature_partition.y
  gini.X = feature_partition.X
  gini.X_argsort = feature_partition.X_argsort
  gini.sample_partition = feature_partition.sample_partition

  gini.n_samples = feature_partition.n_samples
  gini.n_partitions = feature_partition.n_partitions

  -- counts number of samples of respective partition, class which would be left and right of split
  gini.count_left = narray.create({gini.n_partitions, gini.n_classes+1}, narray.int32)
  gini.count_right = narray.create({gini.n_partitions, gini.n_classes+1}, narray.int32)

  -- initialize structure to keep track of hypothetical partitioning
  gini.partition_info = ffi.new(Gini_info, gini.n_partitions)

  -- initialize structure to sample data at one place
  gini.sample_info = ffi.new(Sample_info, gini.n_samples)

  return gini
end

-- executes a full best gini scan
-- for feature f
Gini.full_scan = function(self, f)
  -- reinitialize class counts
  for i = 0, self.n_partitions-1 do
    self.partition_info[i].n_right = self.feature_partition.size.data[i]
    self.partition_info[i].size =  self.partition_info[i].n_right
    self.partition_info[i].position = -1
    self.partition_info[i].n_left = 0
    self.partition_info[i].H_left = 0
    self.partition_info[i].H_right = 0
    self.partition_info[i].best_x = 0
    self.partition_info[i].best_gini = 1e3
    self.partition_info[i].clean = 0
  end
  
  -- reinitiaize count_left and count_right
  self.count_left:assign(0)
  for p = 0, self.n_partitions-1 do
    for c = 0, self.n_classes do
      local class_count = self.feature_partition.partition_class_count:get(p, c)
      self.count_right:set(p, c,  class_count)
      local info = self.partition_info[p]
      if class_count == info.size or info.size < 2 then
        self.partition_info[p].clean = 1
      end
    end
  end

  assert(self.y.strides[0] == 1)
  assert(self.sample_partition.strides[0] == 1)
  
  -- aggregate some sample information
  for i=0, self.n_samples -1 do
    local sample = self.sample_info[i]
    sample.sample = self.X_argsort:get(i,f)
    sample.partition = self.sample_partition.data[sample.sample]
    sample.class = self.y.data[sample.sample]
  end
  
  for i=0, self.n_samples - 1 do
    local sample = self.sample_info[i]

    -- update partition position etc.
    local info = self.partition_info[sample.partition]
    if info.clean ~= 1 and info.n_right > 1 then
      -- update class counts
      local lc = self.count_left:get(sample.partition, sample.class) 
      local lr = self.count_right:get(sample.partition, sample.class) 
      self.count_left:set(sample.partition, sample.class, lc + 1)
      self.count_right:set(sample.partition, sample.class, lr - 1)

      info.position = info.position + 1
      info.n_left = info.n_left + 1
      info.n_right = info.n_right - 1

      -- calculate gini
      info.H_left = info.n_left * info.n_left
      info.H_right = info.n_right * info.n_right

      for c = 0, self.n_classes do
        local count = self.count_left:get(sample.partition, c)
        info.H_left = info.H_left - ( count * count)

        count = self.count_right:get(sample.partition, c) 
        info.H_right = info.H_right - (count * count)
      end

      -- n_left and n_right cannot be zero..
      info.H_left = info.H_left / info.n_left
      info.H_right = info.H_right / info.n_right

      local gini = (info.H_left + info.H_right) / (info.size)

      if gini <= info.best_gini then
        info.best_gini = gini
        info.best_pos = info.position
        info.best_x = self.X:get(sample.sample, f)
        --print("New best", info.best_gini, info.best_pos, info.best_x)
      end
    end
  end
  return self.partition_info
end

