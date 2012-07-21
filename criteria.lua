local math = require("math") 
local ffi = require("ffi")
local bitop = require("bit")

require("luarocks.loader")
local array = require("ljarray.array")
local helpers = require("ljarray.helpers")
local operator = helpers.operator

module(..., package.seeall) -- export all local functions

Gini = {}
Gini.__index = Gini

Gini.create = function(n_classes)
  local gini = {}
  setmetatable(gini, Gini)
  gini.max_n_classes = n_classes
  gini.n_samples = 0
  gini.count_left = helpers.zeros(n_classes)
  gini.count_right = helpers.zeros(n_classes)
  gini.n_left = 0
  gini.n_right = 0
  gini.H_left = 0
  gini.H_right = 0
  gini.position = 0
  return gini
end


Gini.init = function(self, y, start, stop)
  self.y = y
  self.start = start or 0
  self.stop = stop or y.shape[0]

  self.n_samples = stop - start
  self.count_left = helpers.zeros(self.max_n_classes+1)
  self.count_right = helpers.zeros(self.max_n_classes+1)
  self.n_left = 0
  self.n_right = self.n_samples
  -- count classes, initially all samples are to the right
  for i = self.start, self.stop-1 do
    local c = self.y.data[i]
    self.count_right[c] = self.count_right[c] + 1
  end
  self.n_classes = 0
  for c = 0, self.max_n_classes do
    if self.count_right[c] > 0 then
      self.n_classes = self.n_classes + 1
    end
  end
end

Gini.move = function(self, delta)
  local c = self.y.data[self.start + self.n_left]
  self.n_left = self.n_left + 1
  self.n_right = self.n_right - 1
  self.count_left[c] = self.count_left[c] +1
  self.count_right[c] = self.count_right[c] - 1
end

Gini.eval = function(self)
  local H_left = self.n_left * self.n_left
  local H_right = self.n_right * self.n_right

  for c = 0,self.max_n_classes do
    H_left = H_left - (self.count_left[c] *  self.count_left[c])
    H_right = H_right - (self.count_right[c] *  self.count_right[c])
  end

  if self.n_left == 0 then
    H_left = 0
  else
    H_left = H_left / self.n_left
  end
  
  if self.n_right == 0 then
    H_right = 0
  else
    H_right = H_right / self.n_right
  end

  return (H_left + H_right) / self.n_samples
end


