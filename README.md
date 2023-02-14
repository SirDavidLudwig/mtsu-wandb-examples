# Wandb Examples

A repository of Wandb examples to run on MTSU systems.

## Setup

Weights & Biases can typically be installed with:

```bash
pip3 install wandb
```

### Installation on MTSU Systems with Singularity

If you are using Singularity containers on any of the systems (Babbage, Backus, or Hamilton), it can be installed like so using [Dr. Phillips'](https://www.cs.mtsu.edu/~jphillips/) containers:

```bash
singularity exec /home/jphillips/images/csci4850-2023-Spring.sif pip3 install wandb
```

## Run a Job

Jobs can be run on the cluster using the following slurm command:

```bash
sbatch -G 1 -p research ./jobs/<job_script>
```
