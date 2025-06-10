#!/bin/bash
#SBATCH -J parcz-debug
#SBATCH -D /lnet/work/people/stankov/parczech/slurm-logs/
#SBATCH -o create-vocabulary.log
#SBATCH -e create-vocabulary.log
#SBATCH -p cpu-ms
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

conda_path=/lnet/work/people/stankov/miniconda3/bin/conda
export PATH=$conda_path:$PATH
eval "$(conda shell.bash hook)"

conda activate parczech
which python

# Your job command(s)
echo "Running on node: $(hostname)"
echo "Job started at: $(date)"

python /lnet/work/people/stankov/parczech/scripts/create_vocabulary.py