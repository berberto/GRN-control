#!/bin/bash

##SBATCH --partition=cpu
##SBATCH -n 1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=1-20
#SBATCH --job-name=single
#SBATCH --mem=16G
#SBATCH --time=6-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL

parameters=`sed -n "${SLURM_ARRAY_TASK_ID} p" poni-params.txt`

parArray=($parameters)


env=${parArray[0]}
alpha=${parArray[1]}
beta=${parArray[2]}
gamma=${parArray[3]}
wd=${parArray[4]}
seed=${parArray[5]}

echo "env          = $env"
echo "alpha        = $alpha"
echo "beta         = $beta"
echo "gamma        = $gamma"
echo "weight_decay = $wd"
echo "seed         = $seed"
echo " "

WDIR="examples"
cd $WDIR

python -u poni_td3.py  $env  $alpha  $beta  $gamma  $wd  $seed  # 2>&1 | tee -a "logs/$env.log"

