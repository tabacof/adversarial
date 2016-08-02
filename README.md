# Exploring the Space of Adversarial Images

> Tabacof, Pedro and Valle, Eduardo. Exploring the Space of Adversarial Images. arXiv preprint arXiv:1510.05328, 2015. 

Please cite us if you use this code. [ArXiv link](http://arxiv.org/abs/1510.05328)
## Requirements

[Torch7](https://github.com/torch/torch7)

GFortran with BLAS

## L-BFGS-B

The adversarial image optimization problem requires the box-constraints so that the distortions won't make the image go outside the pixel space (RGB = [0, 255]).

For this we use the Fortran library L-BFGS-B written by Nocedal, the author of the algorithm. To compile the library do the following:
```bash
cd lbfgsb
make lib
```
This library is as fast the Torch7 Optim's LBFGS (wihout bound constraints).

## MNIST

For MNIST, the code will train the classifier from scratch. A logistic regression should achieve about 7.5% error, and a standard convolutional network 1%. You need to download the dataset:

```bash
cd mnist
th download.lua
```

## ImageNet

For ImageNet you can use the pre-trained OverFeat network, which is a deep convolutional neural network that won the localization ILSVRC competition in 2013.

First you must download the weights of the network (thanks to Jonghoon Jin):
```bash
cd overfeat
sh install.sh
```

## Adversarial images

Now you can create adversarial images using:
```bash
th adversarial.lua -i image.png
```

Options:
```
-i: image file
-cuda: use GPU support (must have CUDA installed on your computer - test this with require 'cutorch')
-gpu: GPU device number
-ub: unbounded optimization (allow the distortion to go outside the pixel space)
-mc: probe the space around the adversarial image using white noise (default is Gaussian)
-hist: use nonparametric noise instead of Gaussian ("histogram")
-orig: probe the space around the original image instead
-numbermc: number of probes
-mnist: use MNIST instead of ImageNet dataset
-conv: use convolutional network with MNIST (instead of logistic regression)
-itorch: iTorch plotting
-seed: random seed
```

The resulting images and the distortions will be created on the same folder of the image.
