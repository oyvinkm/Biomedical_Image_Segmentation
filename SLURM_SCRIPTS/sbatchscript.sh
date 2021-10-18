#!/bin/bash
#SBATCH--job-name=MyJob
#number of independent tasks we are giong to start in this script
#SBATCH --ntasks=1 --cpus-per-task=10 --mem=12000M
#number of cpus we want to allocate for each program
#SBATCH -p gpu --gres=gpu:titanx:3 --mem=36000M
#We expect that our program should not run longer than 1 hour
#Not that a program will be killed once it exceeds this time
#SBATCH --time=0-10:00:00

echo $CUDA_VISIBLE_DEVICES
time python ../oyvin_code/notebook.py