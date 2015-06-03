--[[
Author: Pedro Tabacof
Contact me through tabacof at gmail dot com
May 2015
License: GPLv3
--]]

require 'torch'
require 'nn'
require 'cunn'

local timer = torch.Timer()

mnist = require 'mnist_utils'

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

if not file_exists('./mnist/mnist model.dat') then
	require 'train'
end

local linLayer = nn.Linear(mnist.n,10)
local softMaxLayer = nn.LogSoftMax()

local model = torch.load('./mnist/mnist model.dat'):float()

local w_train = model:parameters()
local w = linLayer:parameters()
w[1]:copy(w_train[1])
w[2]:copy(w_train[2])

local criterion = nn.ClassNLLCriterion()

disturbanceLayer = nn.Add(mnist.n)
disturbanceLayer.bias:fill(0)

net = nn.Sequential()
net:add(disturbanceLayer)
net:add(linLayer)
net:add(softMaxLayer)

if cuda then net:cuda() end

predict = function()
	if cuda then
		return torch.max(net:forward(rimg_cuda):float(), 1)
	else 
		return torch.max(net:forward(rimg), 1)
	end
end

label = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}

repeat
	local chosen = torch.random() % mnist.test_inputs:size()[1]

	img = mnist.test_inputs[{chosen}]

	rimg, rimg_cuda = img:reshape(mnist.n)
	if cuda then rimg_cuda = img:cuda():reshape(mnist.n) end

	prob, idx = predict()

	print('Prediction:', label[idx:squeeze()], 'Correct:', label[mnist.test_outputs[chosen]])
	
until label[idx:squeeze()] == label[mnist.test_outputs[chosen]]

original_target = idx:squeeze()

mean = mnist.mean
std = mnist.std
n = mnist.n
