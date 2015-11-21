local mnist = {}

mnist.n = 32*32

local train = torch.load('./mnist/mnist.t7/train_32x32.t7', 'ascii')
mnist.train_inputs = train.data:squeeze():float()
mnist.train_outputs = train.labels:float()

local test = torch.load('./mnist/mnist.t7/test_32x32.t7', 'ascii')
mnist.test_inputs = test.data:squeeze():float()
mnist.test_outputs = test.labels:float()

mnist.mean = mnist.train_inputs:mean()
mnist.std = mnist.train_inputs:std()

mnist.train_inputs:add(-mnist.mean)
mnist.train_inputs:mul(1/mnist.std)

mnist.test_inputs:add(-mnist.mean)
mnist.test_inputs:mul(1/mnist.std)

mnist.predictClass = function(model, input_dataset, i)
	model:evaluate()
	local logp = model:forward(input_dataset[i]:reshape(mnist.n))
	local first = {value = -math.huge, pos = 0}
	local second = 0
	for j = 1,(#logp)[1] do
		if logp[j] >= first.value then 
			second = first.pos
			first.value = logp[j]
			first.pos = j
		end
	end
	return first.pos, second
end

mnist.errorRate = function(model, train)
	if train then
		local err1, err2 = 0, 0
		local size = (#mnist.train_inputs)[1]
		for i = 1, size do		
			class1, class2 = mnist.predictClass(model, mnist.train_inputs, i)
			if class1 ~= mnist.train_outputs[i] then
				err1 = err1 + 1
				if class2 ~= mnist.train_outputs[i] then
					err2 = err2 + 1
				end
			end
		end
		print('Train error rate 1-class '.. err1/size*100 .. '% 2-class ' .. err2/size*100.0 .. '%')
	end
	
	local err1, err2 = 0, 0
	local size = (#mnist.test_inputs)[1]
	for i = 1, size do		
		class1, class2 = mnist.predictClass(model, mnist.test_inputs, i)
		if class1 ~= mnist.test_outputs[i] then
			err1 = err1 + 1
			if class2 ~= mnist.test_outputs[i] then
				err2 = err2 + 1
			end
		end
	end
	print('Test error rate 1-class '.. err1/size*100 .. '% 2-class ' .. err2/size*100.0 .. '%')
end

mnist.conv = function(model)
	model:add(nn.Reshape(1, 32, 32))
	model:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
	model:add(nn.ReLU())
	model:add(nn.SpatialMaxPooling(3, 3, 3, 3))
	model:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
	model:add(nn.ReLU())
	model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	model:add(nn.Reshape(64*2*2))
	model:add(nn.Linear(64*2*2, 200))
	model:add(nn.ReLU())
	--model:add(nn.Dropout(0.5))
	model:add(nn.Linear(200, 10))
	model:add(nn.LogSoftMax())
return model

end
return mnist
