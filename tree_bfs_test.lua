-- extend package.path with path of this .lua file:
local filepath = debug.getinfo(1).source:match("@(.*)$") 
local dir = string.gsub(filepath, '/[^/]+$', '') .. "/"
package.path = dir .. "/?.lua;" .. package.path

local math = require("math") 
local ffi = require("ffi")
local bitop = require("bit")

require("luarocks.loader")
local narray = require("ljarray.narray")
local helpers = require("ljarray.helpers")
local tree = require("tree_bfs")
local forest = require("forest_bfs")


local X_train = narray.create({100,2}, narray.float32)
local y_train = narray.create({X_train.shape[0]},narray.int32)

X_train:bind(0,0,50):assign(1)
X_train:bind(0,50,X_train.shape[0]):assign(100)

y_train:bind(0,0,50):assign(1)
y_train:bind(0,50,X_train.shape[0]):assign(2)

local t = tree.create({n_classes = 2})
t:learn(X_train,y_train)

local X_test = X_train

local prediction = t:predict(X_test)
for i = 0, X_test.shape[0]-1 do
  print(prediction:get(i), y_train:get(i))
  assert(prediction:get(i) == y_train:get(i))
end
print("FINISHED TEST1")

local X_train = narray.rand({100,10}, narray.float32)
X_train:add(17)
local y_train = narray.randint(0,2,{X_train.shape[0]},narray.int32):add(1)

print("LEARNING TEST2")
local t = tree.create({n_classes = 2, m_try = X_train.shape[1]})
t:learn(X_train,y_train)

local X_test = X_train

local prediction = t:predict(X_test)
local correct = 0
for i = 0, X_test.shape[0]-1 do
  if  prediction:get(i) == y_train:get(i) then
    correct = correct + 1
  end
  -- assert(prediction:get(i) == y_train:get(i), "prediction failed for index " .. i .. " prediction = " .. prediction:get(i) .. ", GT = "..y_train:get(i))
end
print("TOTAL  CORRECCT COUNT", correct, correct / X_test.shape[0])


print("BEGIN BENCHMARKING")
--math.randomseed(17)


local X_train = narray.rand({10000,10}, narray.float32)
local y_train = narray.randint(0,2,{X_train.shape[0]}):add(1)


local t = forest.create({n_trees = 50, n_classes = 2})

helpers.benchmark(function() t:learn(X_train,y_train) end, 1, "training RF")

local X_test = X_train

helpers.benchmark(function() t:predict(X_test) end, 1, "predicting RF")
local prediction = t:predict(X_test)

local correct = 0
for i = 0, X_test.shape[0]-1 do
  if  prediction:get(i) == y_train:get(i) then
    correct = correct + 1
  end
  -- assert(prediction:get(i) == y_train:get(i), "prediction failed for index " .. i .. " prediction = " .. prediction:get(i) .. ", GT = "..y_train:get(i))
end
print("TOTAL  CORRECCT COUNT", correct, correct / X_test.shape[0])


