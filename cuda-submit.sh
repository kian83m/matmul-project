#!/bin/bash
#SBATCH --account=m4776
#SBATCH -C gpu
#SBATCH --qos=shared
#SBATCH --time=00:01:00
#SBATCH -N 1
#SBATCH -n 1


module load PrgEnv-nvidia cudatoolkit python

make clean 

srun --gres=gpu:1 nsys profile --output=report --stats=true -t nvtx,cuda ./vec_add.x
