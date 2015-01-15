lb = require '../lbfgsb/lbfgsb.lua'

nn = 25

feval = function(x)
	local g = torch.Tensor(nn)
	local f = .25*(x[1]-1.0)*(x[1]-1.0)
	for i = 2, nn do
		local power = x[i] - x[i - 1]*x[i - 1]
		power = power*power
		f = f + power
	end

	local t1 = x[2] - x[1]*x[1];
	g[1] = 2.0*(x[1] - 1.0) - 16.0*x[1]*t1;
	for i = 2, nn - 1 do
		local t2 = t1;
		t1 = x[i + 1]-x[i]*x[i];
		g[i] = 8*t2 - 16.0*x[i]*t1;
	end
	g[nn] = 8.0*t1;

	return 4.0*f, g
end

a = torch.Tensor(nn)
b = torch.Tensor(nn)
bound = torch.IntTensor(nn):fill(2)

for i = 1, nn, 2 do
	a[i] = 1.0;
	b[i] = 100.0;
end

for i = 2, nn, 2 do
	a[i] = -100.0;
	b[i] = 100.0;
end
	
lb.init(nn,5,bound,a,b,1)

x= torch.Tensor(nn):fill(3)

lb.eval(feval, x)
