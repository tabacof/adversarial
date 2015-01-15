----------------------------------------------------------------------
-- example-logistic-regression.lua
--
-- Logistic regression and multinomial logistic regression
--

require 'torch'
require 'nn'
require 'optim'

----------------------------------------------------------------------
-- 1. Create the training data

print('')
print('============================================================')
print('Constructing train_')
print('')

require 'mnist_utils'

n = 32*32

train = torch.load('../mnist.t7/train_32x32.t7', 'ascii')
train__inputs = train.data:squeeze():double()
train__outputs = train.labels:double()

test = torch.load('../mnist.t7/test_32x32.t7', 'ascii')
mnist.test_inputs = test.data:squeeze():double()
mnist.test_outputs = test.labels:double()

mnist.mean = train__inputs:mnist.mean()
mnist.std = train__inputs:mnist.std()

train__inputs:add(-mnist.mean)
train__inputs:mul(1/mnist.std)

mnist.test_inputs:add(-mnist.mean)
mnist.test_inputs:mul(1/mnist.std)

timer = torch.Timer()

mnist.predictClass = function(model, input_train_, i)
	local logp = model:forward(input_train_[i]:reshape(n))
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

mnist.errorRate = function(model)
	local err1, err2 = 0, 0
	for i = 1, (#mnist.test_inputs)[1] do		
		class1, class2 = mnist.predictClass(model, mnist.test_inputs, i)
		if class1 ~= mnist.test_outputs[i] then
			err1 = err1 + 1
			if class2 ~= mnist.test_outputs[i] then
				err2 = err2 + 1
			end
		end
	end
	print('Error rate 1 '.. err1 .. ' 2 ' .. err2)
end

model = nn.Sequential()
--model:add(nn.Linear(n,50))
--model:add(nn.Tanh())
--model:add(nn.Linear(25,10))
--model:add(nn.Tanh())
model:add(nn.Linear(n,10))
model:add(nn.LogSoftMax())

criterion = nn.ClassNLLCriterion()

----------------------------------------------------------------------

x, dl_dx = model:getParameters()

-- The above statement does not create a copy of the parameters in the 
-- model! Instead it create in x and dl_dx a view of the model's weights
-- and derivative wrt the weights. The view is implemented so that when
-- the weights and their derivatives changes, so do the x and dl_dx. The
-- implementation is efficient in that the underlying storage is shared.

-- A note on terminology: In the machine learning literature, the parameters
-- that one seeks to learn are often called weights and denoted with a W.
-- However, in the optimization literature, the parameter one seeks to 
-- optimize is often called x. Hence the use of x and dl_dx above.

-- In the following code, we define a closure, feval, which computes
-- the value of the loss function at a given point x, and the gradient of
-- that function with respect to x. x is the vector of trainable weights,
-- which, in this example, are all the weights of the linear matrix of
-- our mode, plus one bias.

feval = function(x_new)
   -- set x to x_new, if differnt
   -- (in this simple example, x_new will typically always point to x,
   -- so the copy is really useless)
   if x ~= x_new then
      x:copy(x_new)
   end

   -- select a new training sample
   _nidx_ = (_nidx_ or 0) + 1
   if _nidx_ > (#train__inputs)[1] then _nidx_ = 1 end

   local inputs = train__inputs[_nidx_]:reshape(n)
   local target = train__outputs[_nidx_]

   -- reset gradients (gradients are always accumulated, to accomodate 
   -- batch methods)
   dl_dx:zero()

   -- evaluate the loss function and its derivative wrt x, for that sample
   local loss_x = criterion:forward(model:forward(inputs), target)
   model:backward(inputs, criterion:backward(model.output, target))

   -- return loss(x) and dloss/dx
   return loss_x, dl_dx
end

-- Given the function above, we can now easily train the model using SGD.
-- For that, we need to define four key parameters:
--   + a learning rate: the size of the step taken at each stochastic 
--     estimate of the gradient
--   + a weight decay, to regularize the solution (L2 regularization)
--   + a momentum term, to average steps over time
--   + a learning rate decay, to let the algorithm converge more precisely

sgd_params = {
   learningRate = 1e-3,
   learningRateDecay = 1e-4,
   weightDecay = 0,
   momentum = 0
}

-- We're now good to go... all we have left to do is run over the train_
-- for a certain number of iterations, and perform a stochastic update 
-- at each iteration. The number of iterations is found empirically here,
-- but should typically be determinined using cross-validation (i.e.
-- using multiple folds of training/test subsets).

epochs = 5 -- number of times to cycle over our training data

print('')
print('============================================================')
print('Training with SGD')
print('')

for i = 1,epochs do

   -- this variable is used to estimate the average loss
   current_loss = 0

   -- an epoch is a full loop over our training data
   for i = 1,(#train__inputs)[1] do

      -- optim contains several optimization algorithms. 
      -- All of these algorithms assume the same parameters:
      --   + a closure that computes the loss, and its gradient wrt to x, 
      --     given a point x
      --   + a point x
      --   + some parameters, which are algorithm-specific

      w,fs = optim.sgd(feval,x,sgd_params)

      -- Functions in optim all return two things:
      --   + the new x, found by the optimization method (here SGD)
      --   + the value of the loss functions at all points that were used by
      --     the algorithm. SGD only estimates the function once, so
      --     that list just contains one value.

      current_loss = current_loss + fs[1]
   end

   -- report average error on epoch
   current_loss = current_loss / (#train__inputs)[1]
   print('epoch = ' .. i .. ' of ' .. epochs .. ' current loss = ' .. current_loss)

end

mnist.errorRate(model)

print('Time elapsed ' .. timer:time().real .. ' seconds')

--[[
---------------------------------------------------------------------
-- 4.b. Train the model (Using L-BFGS)

-- now that we know how to train the model using simple SGD, we can
-- use more complex optimization heuristics. In the following, we
-- use a second-order method: L-BFGS, which typically yields
-- more accurate results (for linear models), but can be significantly
-- slower. For very large train_s, SGD is typically much faster
-- to converge, and L-FBGS can be used to refine the results.

-- we start again, and reset the trained parameter vector:

--model:reset()

-- next we re-define the closure that evaluates f and df/dx, so that
-- it estimates the true f, and true (exact) df/dx, over the entire
-- train_. This is a full batch approach.
--]]
feval = function(x_new)
   -- set x to x_new, if differnt
   -- (in this simple example, x_new will typically always point to x,
   -- so the copy is really useless)
   if x ~= x_new then
      x:copy(x_new)
   end

   -- reset gradients (gradients are always accumulated, to accomodate 
   -- batch methods)
   dl_dx:zero()

   -- and batch over the whole training train_:
   local loss_x = 0
   for i = 1,(#train__inputs)[1] do
      -- select a new training sample
      _nidx_ = (_nidx_ or 0) + 1
      if _nidx_ > (#train__inputs)[1] then _nidx_ = 1 end

      local inputs = train__inputs[_nidx_]:reshape(n)
      local target = train__outputs[_nidx_]

      -- evaluate the loss function and its derivative wrt x, for that sample
      loss_x = loss_x + criterion:forward(model:forward(inputs), target)
      model:backward(inputs, criterion:backward(model.output, target))
   end

   -- normalize with batch size
   loss_x = loss_x / (#train__inputs)[1]
   dl_dx = dl_dx:div( (#train__inputs)[1] )

   -- return loss(x) and dloss/dx
   return loss_x, dl_dx
end

-- L-BFGS parameters are different than SGD:
--   + a line search: we provide a line search, which aims at
--                    finding the point that minimizes the loss locally
--   + max nb of iterations: the maximum number of iterations for the batch,
--                           which is equivalent to the number of epochs
--                           on the given batch. In that example, it's simple
--                           because the batch is the full train_, but in
--                           some cases, the batch can be a small subset
--                           of the full train_, in which case maxIter
--                           becomes a more subtle parameter.

lbfgs_params = {
   lineSearch = optim.lswolfe,
   maxIter = 50,
   verbose = true
}

print('')
print('============================================================')
print('Training with L-BFGS')
print('')

w,fs = optim.lbfgs(feval,x,lbfgs_params)
-- fs contains all the evaluations of f, during optimization

print('history of L-BFGS evaluations:')
print(fs)

torch.save('logistic.dat', model)

mnist.errorRate(model)

print('Time elapsed ' .. timer:time().real .. ' seconds')
