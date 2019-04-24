#!/bin/bash
#SBATCH --partition=pi_gerstein
#SBATCH --mem 40000
#SBATCH -t 7- #days

source activate mlenv

python dec.py data.csv

