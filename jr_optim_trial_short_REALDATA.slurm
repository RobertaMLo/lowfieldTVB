#!/bin/bash -l
#SBATCH --job-name=EEGo3
#SBATCH --time=168:59:59
#SBATCH --ntasks=21
#SBACTH --cpus-per-task=1
#SBATCH --partition=neuro
#SBATCH --mem-per-cpu=8G

#SBATCH -D /home/neuro_img/Users/Roberta/lowfieldTVB/

source ./tvbenv/bin/activate

#python3 lf_TVB_JR_EEG_optimization_FREEG.py

python3 lf_TVB_JR_EEG_optimization_3_short_REALDATA.py
