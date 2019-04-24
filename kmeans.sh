#!/bin/bash
#SBATCH --partition=pi_gerstein
#SBATCH --mem 40000
#SBATCH -t 7- #days
#SBATCH -c 1
source activate mlenv
python kmeans.py data.csv
