#!/bin/bash
for i in `seq 1 125`;
do
	th adversarial.lua -cuda -mnist -seed $i -mc
done 

mv ./images/mnist/results_from_adversarial.csv ./images/mnist/results_adv.csv 

for i in `seq 1 125`;
do
	th adversarial.lua -cuda -mnist -seed $i -mc -orig
done 

mv ./images/mnist/results_from_original.csv ./images/mnist/results_orig.csv 

for i in `seq 1 125`;
do
	th adversarial.lua -cuda -mnist -seed $i -hist -mc
done 

mv ./images/mnist/results_from_adversarial.csv ./images/mnist/results_adv_hist.csv 

for i in `seq 1 125`;
do
	th adversarial.lua -cuda -mnist -seed $i -hist -mc -orig
done 

mv ./images/mnist/results_from_original.csv ./images/mnist/results_orig_hist.csv 
