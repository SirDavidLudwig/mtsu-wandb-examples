#!/bin/bash
#SBATCH --output=./logs/mnist_dcgan_%j.log

module load singularity

singularity exec                                    \
	--bind ./:/home/jovyan                          \
	--nv                                            \
	/home/jphillips/images/csci4850-2023-Spring.sif \
	python3 ./scripts/mnist_dcgan.py                \
	--epochs 2                                      \
	--mnist_artifact dwl2x/mnist_dataset/mnist:v1
