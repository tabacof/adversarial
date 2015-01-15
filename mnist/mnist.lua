--[[train = torch.load('../mnist.t7/train_32x32.t7', 'ascii')

timer = torch.Timer()
tdata = train.data:double()

for i = 1,32 do
	for j = 1,32 do
		m = tdata[{ {},{1},{i},{j} }]:squeeze():mnist.mean()
		mnist.std = tdata[{ {},{1},{i},{j} }]:squeeze():mnist.std()
		tdata[{ {},{1},{i},{j} }]:add(-m)
		if mnist.std > 0 then
			tdata[{ {},{1},{i},{j} }]:mul(1/mnist.std)
		end
	end
end

print(tdata[{ {1},{1},{},{} }])
]]--

a = torch.rand(2048, 200000)

timer = torch.Timer()

for i = 1, 2048 do
	local s = a[i]:mnist.std()
	local m = a[i]:mnist.mean()
	a[i]:add(-m)
	a[i]:mul(1/s)
end

print('Time elapsed ' .. timer:time().real .. ' seconds')
