# Adversarial images

Reproducing the results of "Intriguing Properties of Neural Networks" and beyond

## Requirements

Torch7 with NN and Image
GFortran with BLAS

## L-BFGS-B

The adversarial image optimization problem requires the box-constraints so that the distortions won't make the image go outside the pixel space (RGB = [0, 255]).

For this we use the Fortran library L-BFGS-B written by Nocedal, the author of the algorithm. To compile the library do the following:
```bash
cd lbfgsb
make lib
```
This library is as fast the Optim's LBFGS (wihout bound constraints).

## MNIST

For MNIST you must train the classifier from scratch.

Under construction.

## ImageNet

For ImageNet you can use the pre-trained OverFeat network, which is deep convolutional neural network that won the ILSVRC competition in 2013.

First you must download the weights of the network:
```bash
cd adversarial
th X.lua
```
  
Now you can create adversarial images using:
```bash
th adversarial.lua image.png
```

Options:
```
-n: number of adversarial images you want to create (default is 1)
-f: fast mode, no optimization used (based on the idea REFNEEDED)
-mc: evaluate the space around the adversarial image using Monte Carlo sampling
-ub: unbounded optimization (allow the distortion to go outside the pixel space)
-cuda: use GPU support (must have CUDA installed on your computer - test this with require 'cutorch')
```

The resulting images and the distortions will be created on the same folder of the image.
