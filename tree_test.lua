-- extend package.path with path of this .lua file:
local filepath = debug.getinfo(1).source:match("@(.*)$") 
local dir = string.gsub(filepath, '/[^/]+$', '') .. "/"
package.path = dir .. "/?.lua;" .. package.path

local math = require("math") 
local ffi = require("ffi")
local bitop = require("bit")

require("luarocks.loader")
local array = require("ljarray.array")
local helpers = require("ljarray.helpers")
local tree = require("tree")
local forest = require("forest")


local X_train = array.create({100,2}, array.float32)
local y_train = array.create({X_train.shape[0]},array.int32)

X_train:bind(0,0,50):assign(1)
X_train:bind(0,50,X_train.shape[0]):assign(100)

y_train:bind(0,0,50):assign(1)
y_train:bind(0,50,X_train.shape[0]):assign(2)

local t = tree.create({n_classes = 2})
t:learn(X_train,y_train)

local X_test = X_train

local prediction = t:predict(X_test)
for i = 0, X_test.shape[0]-1 do
  assert(prediction:get(i) == y_train:get(i))
end
print("FINISHED TEST1")

local X_train = array.rand({100,100}, array.float32)
X_train:add(17)
local y_train = array.randint(0,2,{X_train.shape[0]},array.int32):add(1)

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
math.randomseed(os.time())


local X_train = array.rand({1000000,10}, array.float32)
local y_train = array.randint(0,2,{X_train.shape[0]}):add(1)

local t = forest.create({n_trees = 1, n_classes = 2})

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


