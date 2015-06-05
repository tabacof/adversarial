#!/bin/bash
declare -a arr=("abaya" "ambulance" "banana" "kit_fox" "volcano")
mkdir ~/overfeat_results/

## now loop through the above array
for class in "${arr[@]}"
do
	rm ./images/$class/results.csv

	for i in `seq 1 5`;
	do
		for j in `seq 1 5`;
		do
			th adversarial.lua -cuda -i images/$class/example$i.jpg -seed $j -mc -gpu $1 $2
		done 
	done 
	mkdir ~/overfeat_results/$class
	cp ./images/$class/results.csv ~/overfeat_results/$class
done


