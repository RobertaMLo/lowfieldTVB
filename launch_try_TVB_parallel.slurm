#!/bin/bash -l
#SBATCH --job-name=JHTRY
#SBATCH --time=168:59:59
#SBATCH --ntasks=8
#SBACTH --cpus-per-task=1
#SBATCH --partition=neuro
#SBATCH --mem-per-cpu=8G

#SBATCH -D /home/neuro_img/Users/Roberta/lowfieldTVB/

source ./tvbenv/bin/activate

python try_TVB_parallel.py \
    --pathSC /home/neuro_img/Users/Roberta/LOWFIELD_TVB/HC01/Connectome/SC_dir_hyper.zip \
    --ngen 2 \
    --npop 8 \
    --ncpu 8 \
    --type_acquisition hyper \
    --data /home/neuro_img/Users/Roberta/LOWFIELD_TVB/HC01/EEG/HC01_WB.txt
