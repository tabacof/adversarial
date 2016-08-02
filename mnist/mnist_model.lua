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

if not conv and not file_exists('./mnist/mnist_linear.dat') then
	require 'train'
end

if conv and not file_exists('./mnist/mnist_conv.dat') then
	require 'train'
end

local model
net = nn.Sequential()
disturbanceLayer = nn.Add(mnist.n)
disturbanceLayer.bias:fill(0)
net:add(disturbanceLayer)

if conv then
	model = torch.load('./mnist/mnist_conv.dat'):float()
	local w_train = model:parameters()
	net = mnist.conv(net)
	local w = net:parameters()
	for i = 1, #w_train do
		w[i+1]:copy(w_train[i])
	end
else
	model = torch.load('./mnist/mnist_linear.dat'):float()
	local linLayer = nn.Linear(mnist.n,10)
	local softMaxLayer = nn.LogSoftMax()
	local w_train = model:parameters()
	local w = linLayer:parameters()
	w[1]:copy(w_train[1])
	w[2]:copy(w_train[2])
	net:add(linLayer)
	net:add(softMaxLayer)
end

local criterion = nn.ClassNLLCriterion()

if cuda then net:cuda() end
net:evaluate()
--mnist.errorRate(net)

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
