# Adversarial images

Reproducing the results of "Intriguing Properties of Neural Networks" and beyond

## MNIST

For MNIST you must train the classifier from scratch.

Under construction.

## ImageNet

For ImageNet you can use the pre-trained OverFeat network, which is deep convolutional neural network that won the ILSVRC competition in 2013.

First you must download the weights of the network:
  cd adversarial
  th X.lua
  
Now you can create adversarial images using:
  th adversarial.lua image.png

Options:
-n: number of adversarial images you want to create
-f: fast mode, no optimization used (based on the idea REFNEEDED)
-mc: evaluate the space around the adversarial image using Monte Carlo sampling
-cuda

The resulting images and the distortions will be created on the same folder of the image.
