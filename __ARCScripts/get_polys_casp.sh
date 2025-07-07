#!/bin/bash
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --job-name=casp
#SBATCH --mem-per-cpu=48G
#SBATCH --account=phys-qsum
# #SBATCH --qos=priority
#SBATCH --array=0-3

cd ../

# Define cutoffs array
cutoff_x_vals=(1e-4 1e-8 1e-12 1e-16)
cutoffs=${cutoff_x_vals[$SLURM_ARRAY_TASK_ID]}

/data/phys-qsum/phys1657/spyder-env/bin/python main_LinearLearning.py --cutoff_x $cutoffs --cutoff_y 1e-16 --dataset casp --poly 1 2 3