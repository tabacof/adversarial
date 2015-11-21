#!/bin/bash
for i in `seq 1 125`;
do
	echo th adversarial.lua -mnist -seed $i -mc -conv -cuda
	th adversarial.lua -mnist -seed $i -mc -conv -cuda
done 

mv ./images/mnist/results_from_adversarial.csv ./images/mnist/results_adv.csv 

for i in `seq 1 125`;
do
	echo th adversarial.lua -mnist -seed $i -mc -orig -conv -cuda
	th adversarial.lua -mnist -seed $i -mc -orig -conv -cuda
done 

mv ./images/mnist/results_from_original.csv ./images/mnist/results_orig.csv 

for i in `seq 1 125`;
do
	echo th adversarial.lua -mnist -seed $i -hist -mc -conv -cuda
	th adversarial.lua -mnist -seed $i -hist -mc -conv -cuda
done 

mv ./images/mnist/results_from_adversarial.csv ./images/mnist/results_adv_hist.csv 

for i in `seq 1 125`;
do
	echo th adversarial.lua -mnist -seed $i -hist -mc -orig -conv -cuda
	th adversarial.lua -mnist -seed $i -hist -mc -orig -conv -cuda
done 

mv ./images/mnist/results_from_original.csv ./images/mnist/results_orig_hist.csv 
