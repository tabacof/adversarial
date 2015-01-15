--Set up Git account
----Create SSH server if Torch is too complicated to install
--Train, test and validation error
--Confusion matrix (see Torch7 helper)
--Image statistics: does it heavily depend on the adversarial target choice?
--Image statistics: what is the best distribution to represent it?
----How does it relate to the screen pixels? See Gaussian with variance proportional to pixel intensity
--Problem: Gaussian noise norm is lower than adversarial because of the limits
--How to determine the number of Monte Carlo iterations?
--How to use adversarial images to train any regression (logistic, MLP, conv-net)?
--Profile code to find speed bottlenecks
----Use Shiny!
--Should I iterate over classes to find the one with lowest C? 
----How can this become scalable? 
----How is the target determined in the Intriguing Properties article?
--MNIST: MLP, deep MLP and conv-net
----Should get 1% error rate somewhere
--OverFeat: first reproduce results then use adversarial images to improve classification
----How to incorporate colors?
----Retraining only top layer with adversarial networks
----Retraining all network
----If adversarial images are "very rare", why should they help with generalization? If it does, why is it so?
--Different architectures (Logistic, MLP, Conv-net): how an adversarial image of one affects the other?
----Different initialization has been done but must be repeated as sanity check
--Minus gradient of the original target: why doesn't it work?
--How to approximate MAX(Pj >Pi, ...) on the objective function? How to prove the "Intriguing Properties" approach is one valid approximation for it?

require 'torch'
require 'nn'
--require 'gfx.go'

local timer = torch.Timer()

local mnist = require 'mnist_utils'

local linLayer = nn.Linear(mnist.n,10)
local softMaxLayer = nn.LogSoftMax()

local model = torch.load('logistic.dat')
mnist.errorRate(model)
local w_train = model:parameters()
local w = linLayer:parameters()
w[1]:copy(w_train[1])
w[2]:copy(w_train[2])

local criterion = nn.ClassNLLCriterion()

print('')
print('============================================================')
print('Adversarial training')
print('')

local disturbanceLayer = nn.Add(mnist.n)
disturbanceLayer.bias:fill(0)
print('Disturbance norm before ' .. disturbanceLayer.bias:norm())

local disturbedModel = nn.Sequential()
disturbedModel:add(disturbanceLayer)
disturbedModel:add(linLayer)
disturbedModel:add(softMaxLayer)

--disturbCriterion = nn.DistKLDivCriterion()
local disturbCriterion = nn.ClassNLLCriterion()
local disturb_x, dl_dx = disturbanceLayer:getParameters()

local disturbed_input = 6

local target = mnist.predictClass(disturbedModel, mnist.train_inputs, disturbed_input)
local new_target
repeat new_target = math.random(10) until new_target ~= target
new_target = 1
print("Old target: " .. target .. " New target: " .. new_target)

local lowerBound = -mnist.train_inputs[disturbed_input]:reshape(mnist.n) - mnist.mean/mnist.std
local upperBound = -mnist.train_inputs[disturbed_input]:reshape(mnist.n) + (255 - mnist.mean)/mnist.std

local boundInput = function(x)
	if x < lowerBound[boundCount] then 
		x = lowerBound[boundCount] 
	elseif x > upperBound[boundCount] then
		x = upperBound[boundCount] 
	end
	boundCount = boundCount + 1
	return x 
end

boundCount = 1; disturbanceLayer.bias:apply(boundInput)

mnist.errorRate(disturbedModel)

local feval_disturb = function(x_new)
	if disturb_x ~= x_new then
	  disturb_x:copy(x_new)
	end
		
	-- reset gradients (gradients are always accumulated, to accomodate batch methods)
	dl_dx:zero()

	local inputs = mnist.train_inputs[disturbed_input]:reshape(mnist.n)
	--local target = model:forward(inputs)
	
	-- evaluate the loss function and its derivative wrt x, for that sample
	local loss_x = disturbCriterion:forward(disturbedModel:forward(inputs), new_target)
	loss_x = loss_x + C*disturbanceLayer.bias:norm()^2
	
	disturbedModel:backward(inputs, disturbCriterion:backward(disturbedModel.output, new_target))
	local grad_x = dl_dx[{{1, mnist.n}}] 
	--grad_x:mul(-1)
	grad_x:add(disturb_x*2*C)

	-- return loss(x) and dloss/dx
	return loss_x, grad_x
end

if mnist.predictClass(disturbedModel, mnist.train_inputs, disturbed_input) ~= mnist.train_outputs[disturbed_input] then
	print("This input is already misclassified by the network")
	print("Real: " .. mnist.train_outputs[disturbed_input] .. " Predicted: " .. mnist.predictClass(disturbedModel, mnist.train_inputs, disturbed_input))
else
	local lb = require '../lbfgsb/lbfgsb.lua'
	local bounded = torch.IntTensor(mnist.n):fill(2)
	lb.init(mnist.n, 15, bounded, lowerBound, upperBound, -1)
	local max_iter = 1000

	local tol = 0.001
	local reps = 0
	C = 10
	
	--[[
	repeat
		reps = reps + 1
		C = C - tol
		lb.eval(feval_disturb, disturb_x, max_iter)
	until mnist.train_outputs[disturbed_input] ~= mnist.predictClass(disturbedModel, mnist.train_inputs, disturbed_input)
	--]]
	local C_min, C_max = 0, C
	while C_max - C_min > tol and reps < C/tol do
		reps = reps + 1
		local C_mid = (C_max + C_min)/2
		
		--Beginning evaluation
		if not pred_min or C ~= C_max then
			C = C_min
			lb.eval(feval_disturb, disturb_x, max_iter)
			pred_min = mnist.predictClass(disturbedModel, mnist.train_inputs, disturbed_input)
		end
		
		--Midpoint evaluation
		C = C_mid
		lb.eval(feval_disturb, disturb_x, max_iter)
		local pred_mid = mnist.predictClass(disturbedModel, mnist.train_inputs, disturbed_input)
		
		if pred_mid == pred_min then
			C_min = C
		else
			C_max = C
		end
	end
	C = C_min
	lb.eval(feval_disturb, disturb_x, max_iter)
	--
	
	print("Reps: " .. reps .. " C: " .. C)
	print('Original image: ' .. mnist.train_outputs[disturbed_input] .. ' disturbed image: ' .. mnist.predictClass(disturbedModel, mnist.train_inputs, disturbed_input))
	print('Disturbance norm after ' .. disturbanceLayer.bias:norm())
	
	mnist.errorRate(disturbedModel)
	
	originalDisturbanceLayer = disturbanceLayer.bias:clone()
	local noiseStd = disturbanceLayer.bias:std()
	local noiseMean = disturbanceLayer.bias:mean()
	
	local totalMC = 100

	print('Time elapsed ' .. timer:time().real .. ' seconds')

	print('')
	print('============================================================')
	print('Gaussian noise')
	print('')

	--[[
	local correct = 0
	for i = 1, totalMC do
		disturbanceLayer.bias = torch.Tensor(mnist.n):apply(function(x) return torch.normal(noiseMean, noiseStd) end)
		boundCount = 1; disturbanceLayer.bias:apply(boundInput)
		if mnist.train_outputs[disturbed_input] == mnist.predictClass(disturbedModel, mnist.train_inputs, disturbed_input) then
			correct = correct + 1
		end
	end
	print("Percentage correct from undisturbed: " .. correct/totalMC*100.0 .. "%")
	--]]
	
	correct = 0
	for i = 1, totalMC do
		disturbanceLayer.bias = originalDisturbanceLayer + torch.Tensor(mnist.n):apply(function(x) return torch.normal(0, noiseStd) end)
		boundCount = 1; disturbanceLayer.bias:apply(boundInput)
		if mnist.train_outputs[disturbed_input] == mnist.predictClass(disturbedModel, mnist.train_inputs, disturbed_input) then
			correct = correct + 1
		end
	end
	print("Percentage correct from disturbed: " .. correct/totalMC*100.0 .. "%")
	
	--[[
	print('')
	print('============================================================')
	print('Salt and pepper noise')
	print('')

	local d = 0.1
	saltPepperNoise = function(x)
		saltPepperCount = saltPepperCount + 1
		if torch.uniform() < d then 
			if torch.uniform() < 0.5 then 
				return lowerBound[saltPepperCount]
			else
				return upperBound[saltPepperCount] 
			end
		end
		return 0
	end
				
	correct = 0
	for i = 1, totalMC do
		saltPepperCount = 0; disturbanceLayer.bias = torch.Tensor(mnist.n):apply(saltPepperNoise)
		boundCount = 1; disturbanceLayer.bias:apply(boundInput)
		if mnist.train_outputs[disturbed_input] == mnist.predictClass(disturbedModel, mnist.train_inputs, disturbed_input) then
			correct = correct + 1
		end
	end
	print("Percentage correct from undisturbed: " .. correct/totalMC*100.0 .. "%")
	
	correct = 0
	for i = 1, totalMC do
		saltPepperCount = 0; disturbanceLayer.bias = originalDisturbanceLayer + torch.Tensor(mnist.n):apply(saltPepperNoise)
		boundCount = 1; disturbanceLayer.bias:apply(boundInput)
		if mnist.train_outputs[disturbed_input] == mnist.predictClass(disturbedModel, mnist.train_inputs, disturbed_input) then
			correct = correct + 1
		end
	end
	print("Percentage correct from disturbed: " .. correct/totalMC*100.0 .. "%")
	]]--
	
	local original = mnist.train_inputs[disturbed_input]:clone()
	local disturbed = original + disturbanceLayer.bias:reshape(32,32)
	
	original:mul(mnist.std)
	original:add(mnist.mean)
	
	disturbed:mul(mnist.std)
	disturbed:add(mnist.mean)

	originalDisturbanceLayer:mul(mnist.std)
	originalDisturbanceLayer:add(mnist.mean)
	originalDisturbanceLayer = originalDisturbanceLayer:reshape(32,32)
		
	if gfx then
		gfx.image(original)
		gfx.image(disturbed)
		gfx.image(originalDisturbanceLayer)
	end	
end

print('Time elapsed ' .. timer:time().real .. ' seconds')

