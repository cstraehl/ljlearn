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

local tree = require("tree_bfs")
local criteria = require("criteria_bfs")
local partition = require("partition_bfs")


--module(..., package.seeall) -- export all local functions

local Forest = {}
Forest.__index = Forest

Forest.create = function(options)
  assert(type(options.n_classes) == "number", "Tree: options.n_classes must be set and a number" .. type(options.n_classes))
  local f = {}
  setmetatable(f,Forest)

  f.options = options
  f.n_trees = options.n_trees or 255
  f.max_depth = options.max_depth or -1
  f.criterion_class = options.criterion or criteria.Gini
  f.m_try = options.m_try -- default m_try is determined at learning time
  f.n_classes = options.n_classes

  return f
end

Forest.learn = function(self,X,y)
  assert(X.ndim == 2)
  assert(y.ndim == 1)
  assert(X.shape[0] == y.shape[0])

  self.n_features = X.shape[1]
  self.m_try = self.m_try or math.ceil(math.sqrt(self.n_features)) -- default m_try is sqrt of feature number
  
  -- initialize trees
  self.trees = {}
  for i = 1, self.n_trees do
    local t = tree.create(self.options)
    self.trees[i] = t
  end

  -- if X.order ~= "f" then
  --   X = X:copy("f")
  -- end
  self.X = X
  self.y = y
  self.partition = partition.FeaturePartition.create(X, y, self.n_classes)

  for i = 1, self.n_trees do
    print("training tree", i, "...")
    self.trees[i]:learn(X,y, nil, self.partition)
    print("done.")
  end
end


Forest.predict = function(self,X)
  assert(X.ndim == 2)
  assert(#self.trees >= 1) -- assert forest is trained
  assert(X.shape[1] == self.n_features)

  -- predict with first tree
  local class_counts = self.trees[1]:_predict_class_counts(X)
  
  -- predict with remaining trees
  for t = 2, #self.trees do
    local class_counts2 = self.trees[t]:_predict_class_counts(X)
    assert(class_counts2.shape[0] == class_counts.shape[0])
    assert(class_counts2.shape[1] == class_counts.shape[1])
    
    for i = 0, class_counts.shape[0]-1 do
      for c = 0, class_counts.shape[1]-1 do
        local old_count = class_counts:get(i,c)
        class_counts:set(i,c,old_count + class_counts2:get(i,c))
      end
    end
  end

  local result = narray.create({X.shape[0]},narray.int32)
  result:assign(0) 

  -- get argmax of overall class_counts
  for c = 0, self.n_classes do
    for i = 0, X.shape[0]-1 do
      if class_counts:get(i,c) > class_counts:get(i,result:get(i)) then
        result:set(i,c)
      end
    end
  end

  return result
end

return Forest
