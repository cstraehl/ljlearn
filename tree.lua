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

local criteria = require("criteria")
local partition = require("partition")


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
  tree.node_count = 0

  return tree
end

Tree.learn = function(self,X,y)
  assert(X.ndim == 2)
  assert(y.ndim == 1)
  assert(X.shape[0] == y.shape[0])
  self.n_features = X.shape[1]

  self.m_try = self.m_try or math.ceil(math.sqrt(self.n_features)) -- default m_try is sqrt of feature number

  self.criterion = self.criterion_class.create(self.n_classes)
  self._feature_indices = narray.arange(X.shape[1])
  self:_build(X,y)
  -- print("finished learning")
end

Tree.predict = function(self,X)
  assert(X.ndim == 2)
  assert(X.shape[1] == self.n_features)

  local leaf_nodes = self:_predict_leaf_nodes(X)

  local class_counts = self.leaf_class_count:take_slices(leaf_nodes, 0)

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



Tree._recursive_partition = function(tree,parent_node, is_left_child, partition_number)
   --print("recurse, size: ", tree.partition.size[partition_number])
  
  -- only split further if more then 1 sample
  if tree.partition.size[partition_number] > 1 then
    -- find best split
    local best_split_feat, best_split_val, best_split_x, best_split_pos = tree:_find_best_split(partition_number)

    if best_split_feat ~= -1 then 
      --found split, add split node
      local node = tree:_add_split_node(parent_node, is_left_child, best_split_feat, best_split_x)

      local left_part, right_part = tree.partition:split(partition_number, best_split_feat, best_split_pos)

      -- recurse for left and right partition
      tree:_recursive_partition(node, true, left_part)
      tree:_recursive_partition(node, false, right_part)

    else                                 
      -- did not find possible split, add leaf node
      tree:_add_leaf_node(parent_node, is_left_child)
    end
  else
    -- clean node
    tree:_add_leaf_node(parent_node, is_left_child)
  end
end



Tree._build = function(self, X, y)
  assert(X.ndim == 2)
  assert(y.ndim == 1)
  assert(X.shape[0] == y.shape[0])

  self.node_count = 0

  local crit = self.criterion_class.create(self.n_classes) -- construct criterion
  self.partition = partition.FeaturePartition.create(X)
  self.X = X
  self.y = y
  self.crit = crit
  print("partitioning..")
  self:_recursive_partition(-1, nil, 0)
  print("done")

  -- trim array size
  self:_resize(self.node_count)

  -- setup leaf nodes with class counts
  print("setting up leaf nodes..")
  self:_setup_leaf_nodes(X,y)
  print("done")
  -- print("finished building")
  print("FINISHED")
end

Tree._find_best_split = function(self, partition_number)
-- find best split for self.m_try features
  assert(self._feature_indices.strides[0] == 1)
  self._feature_indices:permute()
  local best_split_val = 1e6
  local best_split_pos = 0
  local best_split_feat = -1
  local best_split_x = 0
  local best_split_x_argsorted = nil
  local best_split_y = nil

  local partition_size = self.partition.size[partition_number]

  local left_partition = nil
  local right_partition = nil

  local x_argsort_i = narray.create({partition_size}, narray.int32)
  local x = narray.create({partition_size}, self.X.dtype)
  local y = narray.create({partition_size}, self.y.dtype)

  
  
  local i = 0

  -- try at least m_try features for splitting
  -- if no valid split point was found, try more features
  while (i < self.X.shape[1] and  best_split_feat == -1) or i < self.m_try do
    local f = self._feature_indices.data[i] -- get feature index to try
    local start = self.partition.start[partition_number]
    local stop = self.partition.stop[partition_number]
    
    for j =0, x.shape[0]-1 do
      x_argsort_i.data[j] = self.partition.X_argsort:get(start + j, f)
      x.data[j] = self.partition.X:get(x_argsort_i.data[j],f)
      y.data[j] = self.y:get(x_argsort_i.data[j])
    end
    
    local split_pos, split_val, split_x = self:_find_best_split_for_feat(x,x_argsort_i,y, self.crit)


    if split_pos ~= -1 and split_val < best_split_val then
      best_split_pos = split_pos
      best_split_val = split_val
      best_split_x = split_x
      best_split_feat = f
    end
    i = i + 1
  end

  -- print("best_split_pos", best_split_pos)
  -- print("best_split_feat", best_split_feat)
  -- print("best_split_x", best_split_x)
  -- print("best_split_val", best_split_val)
  return best_split_feat, best_split_val, best_split_x, best_split_pos 
end

Tree._find_best_split_for_feat = function(self, x, x_argsorted, y, crit)
-- finds the split position that minimizes self.criterion_class
  -- assert(x.ndim == 1)
  -- assert(x.ndim == y.ndim)
  -- assert(x.shape[0] == y.shape[0])
  -- assert(x.strides[0] == 1)
  -- assert(x_argsorted.strides[0] == 1)
  -- assert(y.shape[0]>1)

  crit:init(y)

  local best_split_pos = -1  -- position of split, -1 means invalid
  local best_split_val = crit:eval() -- value of critertion that is minimized
  local best_split_x  -- feature value at best split

  for i = 0, x.shape[0]-2 do
    -- splits can only happen between different feature values
    crit:move(1)
    if x.data[i+1] ~= x.data[i] then
      local crit_val = crit:eval()
      if crit_val < best_split_val then
        best_split_pos = i 
        best_split_val = crit_val
      end
    end
  end
  best_split_x = (x.data[best_split_pos] + x.data[best_split_pos+1])/2
  return best_split_pos, best_split_val, best_split_x
end


Tree._add_split_node = function(self, parent, is_left_child, feature, threshold)
  assert(parent >= -1)
  assert(parent <= self.node_count)
  local id = self.node_count
  self.node_count = self.node_count + 1
  if self.node_count >= self.children.shape[0]-2 then
    self:_resize(self.node_count * 2)
  end

  assert(self.children.shape[0] > id)
  assert(self.split_feature.shape[0] > id)
  assert(self.split_threshold.shape[0] > id)

  if parent >= 0 then
    if is_left_child then
      self.children:set(parent,0,id)
    else
      self.children:set(parent,1,id)
    end
  end

  self.split_feature:set(id,feature)
  self.split_threshold:set(id,threshold)

  return id
end

Tree._add_leaf_node = function(self, parent, is_left_child)
  -- a leaf node is identified by split_feature = -1
  return self:_add_split_node(parent, is_left_child, -1,0)
end

Tree._resize = function(self, size)
  self.children:resize({size,2})
  self.split_feature:resize({size})
  self.split_threshold:resize({size})
end

return Tree
