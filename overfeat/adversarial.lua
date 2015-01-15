--Test with GPU
----Test with CUDNN
--Do Gaussian noise test
--See MNIST todo list
--Warm-starting
----Am I doint it already?
----If I should do it, reuse parameters from close C
----If I shouldn't do it, why?
----Check empirically to see the difference of results (just clear the w tensor)
--LBFGS with multiple evals? What about the "START" string?

require 'nn'
require 'image'
require 'gfx.go'

local ParamBank = require 'ParamBank'
local label     = require 'overfeat_label'

local SpatialConvolution = nn.SpatialConvolution
local SpatialConvolutionMM = nn.SpatialConvolutionMM
local SpatialMaxPooling = nn.SpatialMaxPooling

local cuda = false

if cuda then
	require 'cunn'
	--CUDNN is slower but uses less memory
	--require 'cudnn'
	--SpatialConvolution = cudnn.SpatialConvolution
	--SpatialConvolutionMM = cudnn.SpatialConvolution
	--SpatialMaxPooling = cudnn.SpatialMaxPooling
end

-- OverFeat input arguements
local network  = 'small' or 'big'
network = 'small'
local filename = 'bee.jpg'
timer = torch.Timer()

-- system parameters
local threads = 4
local offset

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(threads)
print('==> #threads:', torch.getnumthreads())

local function nilling(module)
	module.gradBias   = nil
	if module.finput then module.finput = torch.Tensor():typeAs(module.finput) end
	module.gradWeight = nil
	module.output     = torch.Tensor():typeAs(module.output)
	module.fgradInput = nil
	module.gradInput  = nil
end

local function netLighter(network)
	nilling(network)
	if network.modules then
		for _,a in ipairs(network.modules) do
			netLighter(a)
		end
	end
end

--
net = nn.Sequential()
local m = net.modules
if network == 'small' then
	print('==> init a small overfeat network')
	n = 3*231*231
	offset = 2

	disturbanceLayer = nn.Add(n)
	disturbanceLayer.bias:fill(0)

	net:add(disturbanceLayer)
	net:add(nn.View(3, 231, 231))

	if cuda then net:add(SpatialConvolutionMM(3, 96, 11, 11, 4, 4))
	else net:add(SpatialConvolution(3, 96, 11, 11, 4, 4)) end
	net:add(nn.Threshold(0.000001, 0.00000))
	net:add(SpatialMaxPooling(2, 2, 2, 2))
	net:add(SpatialConvolutionMM(96, 256, 5, 5, 1, 1))
	net:add(nn.Threshold(0.000001, 0.00000))
	net:add(SpatialMaxPooling(2, 2, 2, 2))
	net:add(nn.SpatialZeroPadding(1, 1, 1, 1))
	net:add(SpatialConvolutionMM(256, 512, 3, 3, 1, 1))
	net:add(nn.Threshold(0.000001, 0.00000))
	net:add(nn.SpatialZeroPadding(1, 1, 1, 1))
	net:add(SpatialConvolutionMM(512, 1024, 3, 3, 1, 1))
	net:add(nn.Threshold(0.000001, 0.00000))
	net:add(nn.SpatialZeroPadding(1, 1, 1, 1))
	net:add(SpatialConvolutionMM(1024, 1024, 3, 3, 1, 1))
	net:add(nn.Threshold(0.000001, 0.00000))
	net:add(SpatialMaxPooling(2, 2, 2, 2))
	net:add(SpatialConvolutionMM(1024, 3072, 6, 6, 1, 1))
	net:add(nn.Threshold(0.000001, 0.00000))
	net:add(SpatialConvolutionMM(3072, 4096, 1, 1, 1, 1))
	net:add(nn.Threshold(0.000001, 0.00000))
	net:add(SpatialConvolutionMM(4096, 1000, 1, 1, 1, 1))
	net:add(nn.Reshape(1000))
	net:add(nn.LogSoftMax())
	net = net:float()

	-- init file pointer
	print('==> overwrite network parameters with pre-trained weigts')
	ParamBank:init("net_weight_0")
	ParamBank:read(        0, {96,3,11,11},    m[offset+1].weight)
	ParamBank:read(    34848, {96},            m[offset+1].bias)
	ParamBank:read(    34944, {256,96,5,5},    m[offset+4].weight)
	ParamBank:read(   649344, {256},           m[offset+4].bias)
	ParamBank:read(   649600, {512,256,3,3},   m[offset+8].weight)
	ParamBank:read(  1829248, {512},           m[offset+8].bias)
	ParamBank:read(  1829760, {1024,512,3,3},  m[offset+11].weight)
	ParamBank:read(  6548352, {1024},          m[offset+11].bias)
	ParamBank:read(  6549376, {1024,1024,3,3}, m[offset+14].weight)
	ParamBank:read( 15986560, {1024},          m[offset+14].bias)
	ParamBank:read( 15987584, {3072,1024,6,6}, m[offset+17].weight)
	ParamBank:read(129233792, {3072},          m[offset+17].bias)
	ParamBank:read(129236864, {4096,3072,1,1}, m[offset+19].weight)
	ParamBank:read(141819776, {4096},          m[offset+19].bias)
	ParamBank:read(141823872, {1000,4096,1,1}, m[offset+21].weight)
	ParamBank:read(145919872, {1000},          m[offset+21].bias)

elseif network == 'big' then
	print('==> init a big overfeat network')
	n = 3*221*221
	offset = 2
	disturbanceLayer = nn.Add(n)
	disturbanceLayer.bias:fill(0)

	net:add(disturbanceLayer)
	net:add(nn.View(3, 221, 221))
	
	if cuda then net:add(SpatialConvolutionMM(3, 96, 7, 7, 2, 2))
	else net:add(SpatialConvolution(3, 96, 7, 7, 2, 2)) end

	net:add(nn.Threshold(0, 0.000001))
	net:add(SpatialMaxPooling(3, 3, 3, 3))
	net:add(SpatialConvolutionMM(96, 256, 7, 7, 1, 1))
	net:add(nn.Threshold(0, 0.000001))
	net:add(SpatialMaxPooling(2, 2, 2, 2))
	net:add(nn.SpatialZeroPadding(1, 1, 1, 1))
	net:add(SpatialConvolutionMM(256, 512, 3, 3, 1, 1))
	net:add(nn.Threshold(0, 0.000001))
	net:add(nn.SpatialZeroPadding(1, 1, 1, 1))
	net:add(SpatialConvolutionMM(512, 512, 3, 3, 1, 1))
	net:add(nn.Threshold(0, 0.000001))
	net:add(nn.SpatialZeroPadding(1, 1, 1, 1))
	net:add(SpatialConvolutionMM(512, 1024, 3, 3, 1, 1))
	net:add(nn.Threshold(0, 0.000001))
	net:add(nn.SpatialZeroPadding(1, 1, 1, 1))
	net:add(SpatialConvolutionMM(1024, 1024, 3, 3, 1, 1))
	net:add(nn.Threshold(0, 0.000001))
	net:add(SpatialMaxPooling(3, 3, 3, 3))
	net:add(SpatialConvolutionMM(1024, 4096, 5, 5, 1, 1))
	net:add(nn.Threshold(0, 0.000001))
	net:add(SpatialConvolutionMM(4096, 4096, 1, 1, 1, 1))
	net:add(nn.Threshold(0, 0.000001))
	net:add(SpatialConvolutionMM(4096, 1000, 1, 1, 1, 1))
	net:add(nn.Reshape(1000))
	net:add(nn.LogSoftMax())
	net = net:float()

	-- init file pointer
	print('==> overwrite network parameters with pre-trained weigts')
	ParamBank:init("net_weight_1")
	ParamBank:read(        0, {96,3,7,7},      m[offset+1].weight)
	ParamBank:read(    14112, {96},            m[offset+1].bias)
	ParamBank:read(    14208, {256,96,7,7},    m[offset+4].weight)
	ParamBank:read(  1218432, {256},           m[offset+4].bias)
	ParamBank:read(  1218688, {512,256,3,3},   m[offset+8].weight)
	ParamBank:read(  2398336, {512},           m[offset+8].bias)
	ParamBank:read(  2398848, {512,512,3,3},   m[offset+11].weight)
	ParamBank:read(  4758144, {512},           m[offset+11].bias)
	ParamBank:read(  4758656, {1024,512,3,3},  m[offset+14].weight)
	ParamBank:read(  9477248, {1024},          m[offset+14].bias)
	ParamBank:read(  9478272, {1024,1024,3,3}, m[offset+17].weight)
	ParamBank:read( 18915456, {1024},          m[offset+17].bias)
	ParamBank:read( 18916480, {4096,1024,5,5}, m[offset+20].weight)
	ParamBank:read(123774080, {4096},          m[offset+20].bias)
	ParamBank:read(123778176, {4096,4096,1,1}, m[offset+22].weight)
	ParamBank:read(140555392, {4096},          m[offset+22].bias)
	ParamBank:read(140559488, {1000,4096,1,1}, m[offset+24].weight)
	ParamBank:read(144655488, {1000},          m[offset+24].bias)

end
-- close file pointer
ParamBank:close()

--torch.save('overfeat.dat', net)
--local net = torch.load('overfeat.dat')

print('Time elapsed: ' .. timer:time().real .. ' seconds')

if cuda then net:cuda() end

-- load and preprocess image
print('==> prepare an input image')
local img_dim
if network == 'small' then    dim = 231
elseif network == 'big' then  dim = 231 end
local img_raw = image.load(filename):mul(255)
local rh = img_raw:size(2)
local rw = img_raw:size(3)
if rh < rw then
	rw = math.floor(rw / rh * dim)
	rh = dim
else
	rh = math.floor(rh / rw * dim)
	rw = dim
end
img_scale = image.scale(img_raw, rw, rh)

local offsetx = 1
local offsety = 1
if rh < rw then
	offsetx = offsetx + math.floor((rw-dim)/2)
else
	offsety = offsety + math.floor((rh-dim)/2)
end
img = img_scale[{{},{offsety,offsety+dim-1},{offsetx,offsetx+dim-1}}]:floor()

local mean = 118.380948
local std = 61.896913
-- feedforward network
print('==> feed the input image')
timer = torch.Timer()
img:add(-mean):div(std)  -- fixed distn ~ N(118.380948, 61.896913^2)

local rimg, rimg_cuda = img:reshape(n)
if cuda then rimg_cuda = img:cuda():reshape(n) end

local predict = function()
	if cuda then
		return torch.max(net:forward(rimg_cuda):float(), 1)
	else 
		return torch.max(net:forward(rimg), 1)
	end
end
	
prob, idx = predict()

print(label[idx:squeeze()], prob:squeeze())
print('Time elapsed: ' .. timer:time().real .. ' seconds')

print('')
print('============================================================')
print('Adversarial training')
print('')

local disturbCriterion = nn.ClassNLLCriterion()
local disturb_x, dl_dx = disturbanceLayer:getParameters()

if cuda then disturbCriterion:cuda() end

local target = idx:squeeze()
local new_target
new_target = 2

print("Old target: " .. target .. " " .. label[target] .. " New target: " .. new_target .. " " .. label[new_target])

local input, t
if cuda then 
	input = rimg_cuda
	t = torch.CudaTensor(1):fill(new_target)
else 
	input = rimg
	t = new_target
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

local feval_disturb = function(x_new)
	if disturb_x ~= x_new then
	  disturb_x:copy(x_new)
	end
		
	-- reset gradients (gradients are always accumulated, to accomodate batch methods)
	dl_dx:zero()

	-- evaluate the loss function and its derivative wrt x, for that sample
	local out = net:forward(input)
	local loss_x = disturbCriterion:forward(out, t)
	loss_x = loss_x + C*disturbanceLayer.bias:norm()^2
	
	net:backward(input, disturbCriterion:backward(out, t))
	
	-- return loss(x) and dloss/dx
	return loss_x, dl_dx:float():add(x_new*2*C)
end

local lb = require '../lbfgsb/lbfgsb.lua'
local bounded = torch.IntTensor(n):fill(2)
lb.init(n, 7, bounded, lowerBound, upperBound, -1)
local max_iter = 150

local tol = 0.01
local reps = 0
C = 0.01 --This C must be high enough so that the adversarial search fails
local w = disturb_x:float()
--
local C_min, C_max = 0, C

while C_max - C_min > tol and reps < C/tol do
	print("Reps: " .. reps .. " C min: " .. C_min .. " C max: " .. C_max)
	reps = reps + 1
	local C_mid = (C_max + C_min)/2
	
	print("W norm: " .. w:norm())
	--Midpoint evaluation
	C = C_mid
	lb.eval(feval_disturb, w, max_iter)
	local prob, idx = predict()
	local pred_mid = idx:squeeze()

	--w:zero()
	
	if pred_mid == new_target then
		C_min = C
	else
		C_max = C
	end
end
C = C_min
--
lb.eval(feval_disturb, w, max_iter)
prob, idx = predict()

originalDisturbanceLayer = disturbanceLayer.bias:clone():float()

print("Reps: " .. reps .. " C: " .. C)
print('Original image: ' .. label[target] .. ' disturbed image: ' .. label[idx:squeeze()])
print('Disturbance norm after ' .. disturbanceLayer.bias:norm())

original = img:clone()
disturbed = original + originalDisturbanceLayer:reshape(3, 231, 231)

original:mul(std):add(mean)

disturbed:mul(std):add(mean)

originalDisturbanceLayer:mul(std):add(mean)
originalDisturbanceLayer = originalDisturbanceLayer:reshape(3, 231, 231)

print('Time elapsed: ' .. timer:time().real .. ' seconds')

if gfx then
	gfx.image(original)
	gfx.image(disturbed)
	gfx.image(originalDisturbanceLayer)
end	
