------------------------------------------------------------------------
-- This files uses and modifies code from Jonghoon Jin's OverFeat loader
-- See https://github.com/jhjin/overfeat-torch for original code
------------------------------------------------------------------------

-- Harcoded options

cudnn = false
network_size = 'small' or 'big'
network_size = 'small'

-- End options

require 'nn'
require 'image'

print('')
print('============================================================')
print('OverFeat construction')
print('')

local ParamBank = require 'ParamBank'
label     = require 'overfeat_label'

local SpatialConvolution = nn.SpatialConvolution
local SpatialConvolutionMM = nn.SpatialConvolutionMM
local SpatialMaxPooling = nn.SpatialMaxPooling

if cuda then
	if cudnn then
		--CUDNN is slower but uses less memory
		require 'cudnn'
		SpatialConvolution = cudnn.SpatialConvolution
		SpatialConvolutionMM = cudnn.SpatialConvolution
		SpatialMaxPooling = cudnn.SpatialMaxPooling
	else
		require 'cunn'
	end
end

-- Load OverFeat network from into a Torch net
if network_size == 'small' then    dim = 231
elseif network_size == 'big' then  dim = 221 end

local offset
net = nn.Sequential()
local m = net.modules
if network_size == 'small' then
	n = 3*dim*dim
	offset = 2

	disturbanceLayer = nn.Add(n)
	disturbanceLayer.bias:fill(0)

	net:add(disturbanceLayer)
	net:add(nn.View(3, dim, dim))

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
	ParamBank:init("./overfeat/net_weight_0")
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

elseif network_size == 'big' then
	n = 3*dim*dim
	offset = 2
	disturbanceLayer = nn.Add(n)
	disturbanceLayer.bias:fill(0)

	net:add(disturbanceLayer)
	net:add(nn.View(3, dim, dim))
	
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
	ParamBank:init("./overfeat/net_weight_1")
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
local img_raw = image.load(input_image):mul(255)
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

-- Parameters from the training dataset
mean = 118.380948
std = 61.896913

-- Feedforward network
print('==> feed the input image')
img:add(-mean):div(std)  -- fixed distn ~ N(118.380948, 61.896913^2)

rimg, rimg_cuda = img:reshape(n)
if cuda then rimg_cuda = img:cuda():reshape(n) end

predict = function()
	if cuda then
		return torch.max(net:forward(rimg_cuda):float(), 1)
	else 
		return torch.max(net:forward(rimg), 1)
	end
end
	
prob, idx = predict()
original_target = idx:squeeze()

print('Prediction:', label[idx:squeeze()], prob:squeeze())
print('Time elapsed: ' .. timer:time().real .. ' seconds')
