--[[
Author: Pedro Tabacof
Contact me through tabacof at gmail dot com
May 2015
License: GPLv3
--]]

-- Command line options

cmd = torch.CmdLine()
cmd:text('Adversarial network generation')
cmd:text()
cmd:text('Options')
cmd:option('-i', 'none', 'Input image file')
cmd:option('-cuda', false,'CUDA support')
cmd:option('-gpu', 1,'GPU device number')
cmd:option('-ub', false,'Unbounded search')
cmd:option('-mc', false,'Monte Carlo estimation of probability of finding adversarial images by chance')
cmd:option('-numbermc', 100, 'Number of MC distortion samples')
cmd:option('-hist', false, 'Histogram nonparametric noise (resampling with replacement)')
cmd:option('-orig', false, 'MC analysis from original image (instead of adversarial)')
cmd:option('-mnist', false, 'Use MNIST dataset (instead of ImageNet) - train classifier first and save as mnist.dat')
cmd:option('-itorch', false, 'iTorch support for plotting')
cmd:option('-seed', 123,'Random seed')

cmd:text()
params = cmd:parse(arg)

cuda = params['cuda']
deviceNum = params['gpu']
itorch = params['itorch']
unbounded = params['ub']
monte_carlo = params['mc']
hist = params['hist']
mnist = params['mnist']
from_adversarial = not params['orig']
input_image = params['i']
seed = params['seed']
totalMC = params['numbermc']

-- Harcoded options

cudnn = false
threads = 4
warm_start = true
gfx = false
network_size = 'small' or 'big'
network_size = 'small'

-- End options

torch.manualSeed(seed)

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(threads)

require 'nn'
require 'image'
if gfx then require 'gfx.go' end

if cuda then
	require 'cutorch'
	require 'cunn'
	cutorch.setDevice(deviceNum)
end

-- The following required file must define the following global variables:
-- net (sequential model)
-- disturbanceLayer (bias "Add" layer)
-- original_target (original target class)
-- n (input dimension)
-- mean (input mean)
-- std (input std)
-- label (string table)
-- rimg (original image)
-- predict (function)

package.path = package.path .. ';./mnist/?.lua;./overfeat/?.lua;./lbfgsb/?.lua'

timer = torch.Timer()

if mnist then
	if input_image == 'none' then input_image = 'images/mnist/last_mnist.jpg' end
	require 'mnist'
else
	if input_image == 'none' then input_image = 'images/bee/bee.jpg' end
	require 'overfeat'
end

print('')
print('============================================================')
print('Adversarial training')
print('')

last_time = timer:time().real

local disturbCriterion = nn.ClassNLLCriterion()
local disturb_x, dl_dx = disturbanceLayer:getParameters()

if cuda then disturbCriterion:cuda() end

local adversarial_target = original_target 

-- Find a new label randomly
while original_target == adversarial_target do 
	adversarial_target = torch.random() % #label + 1
end

print("Original prediction: #" .. original_target .. " " .. label[original_target])
print("Adversarial: #" .. adversarial_target .. " " .. label[adversarial_target])

local input, target
if cuda then 
	input = rimg_cuda
	target = torch.CudaTensor(1):fill(adversarial_target)
else 
	input = rimg
	target = adversarial_target
end

local lowerBound = -rimg - mean/std
local upperBound = -rimg + (255 - mean)/std

local boundInput = function(x)
	if x < lowerBound[boundCount] then 
		x = lowerBound[boundCount] 
	elseif x > upperBound[boundCount] then
		x = upperBound[boundCount] 
	end
	boundCount = boundCount + 1
	return x 
end

local sign = function(x)
	if x < 0 then
		x = -1
	elseif x > 0 then
		x = 1
	end
	return x
end


local lbfgsb_function_gradient = function(x_new)
	if disturb_x ~= x_new then
	  disturb_x:copy(x_new)
	end
		
	-- Reset gradients (gradients are always accumulated, to accomodate batch methods)
	dl_dx:zero()

	-- Evaluate the loss function and its derivative wrt x, for that sample
	local out = net:forward(input)
	local loss_x = disturbCriterion:forward(out, target)
	-- Add regularization term
	-- L2 regularization is default, uncomment lines below for L1 regularization
	loss_x = loss_x + C*disturbanceLayer.bias:norm()^2
	--local l1 = disturbanceLayer.bias:clone()
	--loss_x = loss_x + C*l1:abs():sum()
			
	net:backward(input, disturbCriterion:backward(out, target))
	local gradient = dl_dx:float():add(x_new*2*C)
	--local l1 = x_new:clone()
	--local gradient = dl_dx:float():add(l1:sign()*C)

	-- Return loss and gradient
	return loss_x, gradient
end

local lb = require 'lbfgsb'
local bounded
if unbounded then
	bounds = torch.IntTensor(n):fill(0)
else
	bounds = torch.IntTensor(n):fill(2)
end

lb.init(n, 15, bounds, lowerBound, upperBound, -1)
local lbfgsb_max_iter = 150
local lbfgsb_w = disturb_x:float()

print("Finding starting point")
local C_init = 0.01

--The initial C must be high enough so that the adversarial search will fail
repeat 
	print("Trying C = ", C_init)
	C = C_init
	lb.eval(lbfgsb_function_gradient, lbfgsb_w, lbfgsb_max_iter)
	local prob, idx = predict()
	local pred_init = idx:squeeze()
	print("Prediction: #" .. pred_init, label[pred_init])
	C_init = C_init*10.0
until pred_init == original_target
C_init = C_init/10.0

local C_min, C_max = 0, C_init
local reps = 0
local tol = C_init/10.0

while C_max - C_min > tol do
	print("Reps: " .. reps .. " C min: " .. C_min .. " C max: " .. C_max)
	reps = reps + 1
	local C_mid = (C_max + C_min)/2
	
	print("Disturbance norm: " .. lbfgsb_w:norm())
	--Midpoint evaluation
	C = C_mid
	lb.eval(lbfgsb_function_gradient, lbfgsb_w, lbfgsb_max_iter)
	local prob, idx = predict()
	local pred_mid = idx:squeeze()
	print("Prediction: #" .. pred_mid, label[pred_mid])
	
	if not warm_start then
		lbfgsb_w:zero()
	end
	
	if pred_mid == adversarial_target then
		C_min = C
	else
		C_max = C
	end
end
C = C_min
lb.eval(lbfgsb_function_gradient, lbfgsb_w, max_iter)
prob, idx = predict()
print("Reps: " .. reps .. " C: " .. C)

originalDisturbanceLayer = disturbanceLayer.bias:clone():float()

print('Disturbance norm ' .. originalDisturbanceLayer:norm())
print('')
print('Original prediction: ' .. label[original_target])
print('New prediction: ' .. label[idx:squeeze()])
if original_target == idx:squeeze() then
	print('Adversarial image search failed!')
	print('Try increasing the initial max C or lower the tol value')
end

if unbounded then
	local outOfBounds, inNorm, outNorm = 0, 0, 0
	for i = 1, n do	
		if originalDisturbanceLayer[i] < lowerBound[i] or originalDisturbanceLayer[i] > upperBound[i] then
			outOfBounds = outOfBounds + 1
			outNorm = outNorm + originalDisturbanceLayer[i]^2
		else
			inNorm = inNorm + originalDisturbanceLayer[i]^2
		end
	end
	print("Out of bounds disturbances (%):", 100.0*outOfBounds/n)
	print("Out of bounds squared norm:", outNorm)
	print("In bounds squared norm:", inNorm)
	print("Proportion (%):", 100.0*outNorm/(inNorm+outNorm))
	--boundCount = 1; originalDisturbanceLayer:apply(boundInput)
end

print('Time elapsed: ' .. timer:time().real - last_time .. ' seconds')
last_time = timer:time().real

mc_stats = ''
max_mc = 0

if monte_carlo then
	print('')
	print('============================================================')
	print('Monte Carlo analysis')
	print('')

	local wilsonInterval = function(p, n)
		z = 1.96 -- 95% confidence 
		local lower = ((p + z*z/(2*n) - z * math.sqrt((p*(1-p)+z*z/(4*n))/n))/(1+z*z/n))
		local upper = ((p + z*z/(2*n) + z * math.sqrt((p*(1-p)+z*z/(4*n))/n))/(1+z*z/n))
		return lower, upper
	end
	
	local noiseStd = originalDisturbanceLayer:std()
	local noiseMean = originalDisturbanceLayer:mean()
	
	max_mc = hist and 0 or 5 -- For non-parametric sampling, do only 1 iteration
	
	for i = -max_mc, max_mc do
		collectgarbage() -- Prevent GPU being out of memory
		original_count = 0
		adversarial_count = 0
		local actualNoiseStd = 0
		for j = 1, totalMC do
			local randomDisturbanceLayer
			if hist then
				randomDisturbanceLayer = torch.Tensor(n):apply(function(x) return originalDisturbanceLayer[math.random(1, n)] end)
			else
				randomDisturbanceLayer = torch.Tensor(n):apply(function(x) return torch.normal(noiseMean, noiseStd*2^i) end)
			end
			
			if from_adversarial then
				randomDisturbanceLayer = randomDisturbanceLayer + originalDisturbanceLayer
			end
			
			boundCount = 1; randomDisturbanceLayer:apply(boundInput)
			actualNoiseStd = actualNoiseStd + (randomDisturbanceLayer - originalDisturbanceLayer):std()

			if cuda then randomDisturbanceLayer = randomDisturbanceLayer:cuda() end
			
			disturbanceLayer.bias = randomDisturbanceLayer
			
			local prob, idx = predict()
			--print(original_target .. ' ' .. idx:squeeze())
			if original_target == idx:squeeze() then
				original_count = original_count + 1
			elseif adversarial_target == idx:squeeze() then
				adversarial_count = adversarial_count + 1
			end
		end
		local p_original = original_count/totalMC
		local p_adversarial = adversarial_count/totalMC
		local p_lower, p_upper =  wilsonInterval(p_original, totalMC)
		print("Percentage original:", p_original*100.0 .. "%")
		print("with 95% confidence Wilson score interval [" .. p_lower .. ", " .. p_upper .. "]")
		print("Percentage adversarial:", p_adversarial*100.0 .. "%")
		print("Targe noise std: ".. noiseStd .. " - actual noise std average " .. actualNoiseStd/totalMC)
		mc_stats = mc_stats .. noiseStd .. ';' .. actualNoiseStd/totalMC .. ';' .. p_original*100.0 .. ';' .. p_adversarial*100.0 .. ';'

		noiseStd = noiseStd * 0.9
	end
end

original = img:clone()
if mnist then
	disturbed = original + originalDisturbanceLayer:reshape(32, 32)
else
	disturbed = original + originalDisturbanceLayer:reshape(3, 231, 231)
end

original:mul(std):add(mean)
disturbed:mul(std):add(mean)

originalDisturbanceLayer:mul(std):add(mean)

if itorch then
	Plot = require 'itorch.Plot'
	plot = Plot():histogram(originalDisturbanceLayer):draw()
	plot:title('Disturbance histogram'):redraw()
end

function kurtosis(t)
	local m = t:mean()
	local sqSum = 0
	local fourthSum = 0
	local n = t:size()[1]
	for i = 1, n do
		local sq = (t[i] - m)*(t[i] - m)
		sqSum = sqSum + sq
		fourthSum = fourthSum + sq*sq
	end
	return n*fourthSum/(sqSum*sqSum) - 3
end

function skewness(t)
	local m = t:mean()
	local sqSum = 0
	local thirdSum = 0
	local n = t:size()[1]
	for i = 1, n do
		sqSum = sqSum + (t[i] - m)*(t[i] - m)
		thirdSum = thirdSum + (t[i] - m)*(t[i] - m)*(t[i] - m)
	end
	return (thirdSum/n)/math.pow(sqSum/(n - 1), 3/2)
end

local adv_mean = originalDisturbanceLayer:mean()
local adv_std = originalDisturbanceLayer:std()
local adv_kurt = kurtosis(originalDisturbanceLayer)
local adv_skew = skewness(originalDisturbanceLayer)

print("Disturbance statistics")
print("Mean", adv_mean)
print("Std", adv_std)
print("Kurt", adv_kurt)
print("Skew", adv_skew)

local csv_stats = label[original_target] .. ';' .. label[adversarial_target] .. ';' .. adv_mean .. ';' .. adv_std .. ';' .. adv_kurt .. ';' .. adv_skew .. ';'

if mnist then
	originalDisturbanceLayer = originalDisturbanceLayer:reshape(32, 32)
else
	originalDisturbanceLayer = originalDisturbanceLayer:reshape(3, 231, 231)
end

print('Time elapsed: ' .. timer:time().real - last_time  .. ' seconds')

save_name = input_image:gsub(".jpg", "")
save_name = save_name:gsub(".png", "")

if itorch then
	plot:save(save_name  .. '_distortion_hist_' .. label[adversarial_target] .. '.html')
end

image.save(save_name .. '_adversarial_'  .. label[adversarial_target] .. '.png', disturbed/255)
image.save(save_name  .. '_distortion_'  .. label[adversarial_target] .. '.png', originalDisturbanceLayer/255)
image.save(save_name  .. '_original'  .. label[adversarial_target] .. '.png', original/255)

while save_name:sub(#save_name, #save_name) ~= '/' do
	save_name = save_name:sub(1, #save_name - 1)
end

if from_adversarial then
	file = io.open(save_name .. '/results_from_adversarial.csv', 'a')
else
	file = io.open(save_name .. '/results_from_original.csv', 'a')
end

io.output(file)
io.write('Original label; Adversarial label; Disturbance mean; Std; Kurtosis; Skewness;')

if hist then io.write('Expected MC noise std; Actual noise std; Original (%); Adversarial (%);') end

if max_mc > 0 then
	for i = -max_mc, max_mc do 
		io.write('Expected MC noise std; Actual noise std; Original (%); Adversarial (%);')
	end
end
io.write('\n' .. csv_stats .. mc_stats .. '\n')
io.close(file)

if gfx then
	gfx.image(original)
	gfx.image(disturbed)
	gfx.image(originalDisturbanceLayer)
end	
