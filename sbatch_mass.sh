#!/bin/bash
rm *.out
sbatch autoencoder.sh
sbatch cnn.sh
sbatch dec.sh
sbatch kmeans.sh
sbatch random_forest.sh
sbatch svc.sh
