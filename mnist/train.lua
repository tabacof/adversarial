------------------------------------------------------------------------
-- Based on the code by Ronan Collobert
-- https://github.com/andresy/torch-demos/blob/master/logistic-regression/example-logistic-regression.lua
------------------------------------------------------------------------

require 'torch'
require 'nn'
require 'optim'
require 'cunn'
----------------------------------------------------------------------
-- 1. Create the training data
----------------------------------------------------------------------

print('')
print('============================================================')
print('Constructing training data')
print('')

timer = torch.Timer()

torch.setdefaulttensortype('torch.FloatTensor')

package.path = package.path .. ';./mnist/?.lua;'

local mnist = require 'mnist_utils'

model = nn.Sequential()

-- ConvNet that achieves < 1% error rate
if conv then
	model = mnist.conv(model)
-- Softmax regression achieves ~7.5% error rate
else
	model:add(nn.Linear(mnist.n, 10))
	model:add(nn.LogSoftMax())
end

model:cuda()

criterion = nn.ClassNLLCriterion()
criterion:cuda()
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
   if _nidx_ > (#mnist.train_inputs)[1] then _nidx_ = 1 end

   local inputs = mnist.train_inputs[_nidx_]:reshape(mnist.n)
   local target = mnist.train_outputs[_nidx_]

   -- reset gradients (gradients are always accumulated, to accomodate 
   -- batch methods)
   dl_dx:zero()

   -- evaluate the loss function and its derivative wrt x, for that sample
   local loss_x = criterion:forward(model:forward(inputs:cuda()), target)
   model:backward(inputs:cuda(), criterion:backward(model.output, target))

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
   learningRate = 1e-4,
   learningRateDecay = 5e-7,
   weightDecay = 1e-3,
   momentum = 0.9
}

-- We're now good to go... all we have left to do is run over the train_
-- for a certain number of iterations, and perform a stochastic update 
-- at each iteration. The number of iterations is found empirically here,
-- but should typically be determinined using cross-validation (i.e.
-- using multiple folds of training/test subsets).

epochs = 100 -- number of times to cycle over our training data

print('')
print('============================================================')
print('Training with SGD')
print('')

model:training()

for i = 1,epochs do

   -- this variable is used to estimate the average loss
   current_loss = 0

   -- an epoch is a full loop over our training data
   for i = 1,(#mnist.train_inputs)[1] do

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
   current_loss = current_loss / (#mnist.train_inputs)[1]
   print('epoch = ' .. i .. ' of ' .. epochs .. ' current loss = ' .. current_loss)

end

if conv then
	torch.save('./mnist/mnist_conv.dat', model:float())
else
	torch.save('./mnist/mnist_linear.dat', model:float())
end
mnist.errorRate(model)

print('Time elapsed ' .. timer:time().real .. ' seconds')
