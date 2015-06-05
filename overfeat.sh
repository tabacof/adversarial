#!/bin/bash
declare -a arr=("abaya" "ambulance" "banana" "kit_fox" "volcano")
mkdir ~/overfeat_results/

## now loop through the above array
for class in "${arr[@]}"
do
	for i in {1..5}
	do
		for j in {1..5}
		do
			th adversarial.lua -cuda -i images/$class/example$i.jpg -seed $j -mc -gpu $1 $2
		done 
	done 
	cp -r ./images/$class/ ~/overfeat_results/
done


