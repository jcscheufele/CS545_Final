#!/bin/sh
#SBACTH --job-name=“CS545 Final”
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100
#SBATCH --time=0-10:00
#SBATCH --mem=160gb
#SBATCH --partition=short
#SBATCH --mail-user=jcscheufele@wpi.edu
#SBATCH --mail-type=ALL
eval "$(conda shell.bash hook)"
conda activate dip2
srun python run_obj_detector.py