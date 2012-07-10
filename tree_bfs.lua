-- extend package.path with path of this .lua file:local filepath = debug.getinfo(1).source:match("@(.*)$") 
local filepath = debug.getinfo(1).source:match("@(.*)$") 
local dir = string.gsub(filepath, '/[^/]+$', '') .. "/"
package.path = dir .. "/?.lua;" .. package.path

local math = require("math") 
local ffi = require("ffi")
local bitop = require("bit")

require("luarocks.loader")
local narray = require("ljarray.narray")
local helpers = require("ljarray.helpers")
local operator = helpers.operator

local criteria = require("criteria_bfs")
local partition = require("partition_bfs")


--module(..., package.seeall) -- export all local functions

local Tree = {}
Tree.__index = Tree


Tree.create = function(options)
  assert(type(options.n_classes) == "number", "Tree: options.n_classes must be set and a number" .. type(options.n_classes))
  local tree = {}
  setmetatable(tree, Tree)
  tree.n_classes = options.n_classes
  tree.max_depth = options.max_depth or -1
  tree.criterion_class = options.criterion or criteria.Gini
  tree.m_try = options.m_try -- default m_try is determined at learning time

  tree.children = narray.create({2048,2})
  tree.split_feature = narray.create({2048})
  tree.split_threshold = narray.create({2048})
  tree.node_count = 1

  return tree
end

Tree.learn = function(self,X,y, sample_mask, partition)
  assert(X.ndim == 2)
  assert(y.ndim == 1)
  assert(X.shape[0] == y.shape[0])
  self.n_features = X.shape[1]

  self.m_try = self.m_try or math.ceil(math.sqrt(self.n_features)) -- default m_try is sqrt of feature number

  self.sample_mask = sample_mask
  self.partition = partition

  self:_build(X,y)

  self.sample_mask = nil
  self.partition = nil
  -- print("finished learning")
end

Tree.predict = function(self,X)
  assert(X.ndim == 2)
  assert(X.shape[1] == self.n_features)

  local class_counts = self:_predict_class_counts(X)

  local result = narray.create({X.shape[0]},narray.int32)
  result:assign(0) 

  for c = 0, self.n_classes do
    for i = 0, X.shape[0]-1 do
      if class_counts:get(i,c) > class_counts:get(i,result:get(i)) then
        result:set(i,c)
      end
    end
  end

  return result
end

Tree._predict_class_counts = function(self, X)
  assert(X.ndim == 2)
  assert(X.shape[1] == self.n_features)

  local leaf_nodes = self:_predict_leaf_nodes(X)

  local class_counts = self.leaf_class_count:take_slices(leaf_nodes, 0)
  return class_counts
end


Tree._predict_leaf_nodes = function(self, X)
  local leaf_node_indices = narray.create({X.shape[0]}, narray.int32) 
  for i=0,X.shape[0]-1 do
    local node = 0
    local f = self.split_feature:get(node)
    local did = false
    while f ~= -1 or did == false do
      did = true
      if X:get(i, f) <= self.split_threshold:get(node) then
        node = self.children:get(node, 0) --left branch
      else
        node = self.children:get(node, 1) --right branch
      end
      f = self.split_feature:get(node)
    end
    leaf_node_indices:set(i,node)
    -- -- recursively traverse tree, starting from root node 0
    -- leaf_node_indices:set(i, _predict_recurse(self,X,i,0))
  end
  return leaf_node_indices
end


Tree._setup_leaf_nodes = function(self, X, y)
  assert(X.shape[0] == y.shape[0])

  local leaf_class_count = narray.create({self.node_count+1, self.n_classes+1}, narray.int32)
  leaf_class_count:assign(0)

  local leaf_node_indices = self:_predict_leaf_nodes(X)
  
  for i=0,X.shape[0]-1 do
    local node = leaf_node_indices:get(i)
    local class = y:get(i)
    local count = leaf_class_count:get(node,class) + 1
    leaf_class_count:set(node,class,count)
  end

  self.leaf_class_count = leaf_class_count
end


Tree._build = function(self, X, y)
  assert(X.ndim == 2)
  assert(y.ndim == 1)
  assert(X.shape[0] == y.shape[0])

  self.node_count = 1
  
  self._feature_indices = narray.arange(0,X.shape[1])

  if self.partition == nil then
    self.partition = partition.FeaturePartition.create(X, y, self.n_classes)
  else
    self.partition:reset() -- reinitialize partition
  end

  self.X = X
  self.y = y

  self:_iterative_partition()

  -- trim array size
  self:_resize(self.node_count)

  -- setup leaf nodes with class counts
  self:_setup_leaf_nodes(X,y)
  -- print("finished building")
end

Tree._iterative_partition = function(self)
  local did_split = true
  local level = 0
  local max_level = 200

  -- presort for now
  for f = 0, self.X.shape[1]-1 do 
      self.partition:ensure_sorted(f)
  end

  while did_split and level < max_level do
    did_split = false
    --print("building tree level", level)
    level = level + 1
    self._feature_indices:permute()

    local crit = self.criterion_class.create(self.partition)

    -- allocate and initialize best criteria info array for parition
    local best_info = ffi.new(crit.gini_info_t, self.partition.n_partitions)
    for p = 0,self.partition.n_partitions-1 do
      best_info[p].best_gini = 1e3
    end

    local best_feat = narray.create({self.partition.n_partitions}, narray.int32)
    best_feat:assign(-1)

    -- find best split amongst m_try features for all leaf nodes
    for t = 0, self.m_try-1 do
      local f = self._feature_indices:get(t)
      self.partition:ensure_sorted(f)

      -- find best split for current feature for all current leafes
      local gini_info = crit:full_scan(f)

      -- test wether current feature gini is better 
      -- then the current best one for each partition
      for p = 0, self.partition.n_partitions-1 do
        local info = gini_info[p]
        if info.best_pos ~= -1 and info.best_gini < best_info[p].best_gini then
          -- print("best gini", info.best_gini)
          best_info[p] = info
          best_feat.data[p] = f
        end
      end
    end

    local splits = narray.create({self.partition.n_partitions, 2}, narray.int32)
    splits:assign(-1)

    -- split the nodes, keep track of splits
    for p = 0, self.partition.n_partitions-1 do
      local feat = best_feat.data[p]
      if feat ~= -1 then
        did_split = true
        local p_info = best_info[p]
        -- print("split pos", p, p_info.best_pos, self.partition.size:get(p))
        -- split node
        local left, right = self:_split_node(p, feat, p_info.best_x)

        -- keep track of split
        splits:set(p,0,left)
        splits:set(p,1,right)
      end
    end

    local new_sample_partitions = narray.create({self.partition.n_samples}, narray.int32)
    --  initialize with old partition values
    new_sample_partitions:assign(self.partition.sample_partition)

    -- reassign all the samples to their new partitions
    for s = 0, self.partition.n_samples-1 do
      local p = self.partition.sample_partition:get(s) --get current partitin of sample
      local info = best_info[p]
      local feat = best_feat.data[p]
      -- reassign partition if a valid split was found
      if  feat ~= -1 then
        local left, right = splits:get(p,0), splits:get(p,1)
        if self.X:get(s,feat) <= info.best_x then
          new_sample_partitions:set(s, left)
        else
          new_sample_partitions:set(s, right)
        end
      end
    end

    -- update partition manager
    self.partition:update(new_sample_partitions)
  end

end


Tree._split_node = function(self, node, feature, threshold)
  assert(node >= -1)
  assert(node <= self.node_count)
  local id_left = self.node_count
  local id_right = self.node_count +1
  self.node_count = self.node_count + 2
  if self.node_count >= self.children.shape[0]-2 then
    self:_resize(self.node_count * 2)
  end

  assert(self.children.shape[0] > node)
  assert(self.split_feature.shape[0] > node)
  assert(self.split_threshold.shape[0] > node)

  self.children:set(node,0,id_left)
  self.children:set(node,1,id_right)

  self.split_feature:set(node,feature)
  self.split_threshold:set(node,threshold)
  
  -- initialize nodes as leaf-nodes
  -- which are identified by feature = -1
  self.split_feature:set(id_left,-1)
  self.split_feature:set(id_right,-1)

  return id_left, id_right
end

Tree._resize = function(self, size)
  self.children:resize({size,2})
  self.split_feature:resize({size})
  self.split_threshold:resize({size})
end

return Tree      
