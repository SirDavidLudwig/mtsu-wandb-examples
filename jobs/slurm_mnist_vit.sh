#!/bin/bash
#SBATCH --output=./logs/mnist_vit_%j.log

module load singularity

singularity exec                                    \
	--nv                                            \
	--bind ./:/home/jovyan                          \
	/home/jphillips/images/csci4850-2023-Spring.sif \
	python3 ./scripts/mnist_vit.py                  \
	--epochs 2
