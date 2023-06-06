#!/bin/bash

#SBATCH --partition=cpu
#SBATCH -n 1
##SBATCH --partition=gpu
##SBATCH --gres=gpu:1
#SBATCH --array=1-6
#SBATCH --job-name=pattern
#SBATCH --mem=64G
#SBATCH --time=21-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL

parameters=`sed -n "${SLURM_ARRAY_TASK_ID} p" poni-pattern-params.txt`
parArray=($parameters)

env=${parArray[0]}
n_agents=${parArray[1]}
kappa=${parArray[2]}
size=${parArray[3]}
lam=${parArray[4]}
d_memory=${parArray[5]}
wd=${parArray[6]}
seed=${parArray[7]}

echo "ENV      = $env"
echo "n_agents = $n_agents"
echo "kappa    = $kappa"
echo "size     = $size"
echo "lam      = $lam"
echo "d_memory = $d_memory"
echo "seed     = $seed"

WDIR="examples"
cd $WDIR

python -u poni_td3_pattern.py  $env  $n_agents  $kappa  $size  $lam  $d_memory  $wd  $seed
