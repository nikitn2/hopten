#!/bin/bash
#SBATCH --partition=short
#SBATCH -n 8
#SBATCH --time=10:00:00
#SBATCH --job-name=build_spyder-env
#SBATCH --account=phys-qsum
##SBATCH --qos=priority



module load Anaconda3
export CONPREFIX=$DATA/spyder-env
conda create --prefix $CONPREFIX -y
source activate $DATA/spyder-env

conda install -c conda-forge python quimb scikit-learn pandas matplotlib seaborn -y
pip install -U jax[cuda12]
pip install tensorflow
pip install tn4ml