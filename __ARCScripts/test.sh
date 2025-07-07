#!/bin/bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --job-name=mnist_p2_test
#SBATCH --mem-per-cpu=32G
#SBATCH --account=phys-qsum
#SBATCH --qos=priority

cd ../


/data/phys-qsum/phys1657/spyder-env/bin/python main_LinearLearning.py --cutoff_x 1e-16 --cutoff_y 1e-8 --poly 2 --dataset mnist --classify True