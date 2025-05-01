#!/bin/bash
#SBATCH --job-name=clean-express
#SBATCH -D /lnet/work/people/stankov/parczech/slurm-logs/migrate
#SBATCH -o clean-express.log
#SBATCH -e clean-express.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G

echo "Start cleaning at $(date)"

rm -rf /lnet/express/work/people/stankov/{*,.*} 

echo "Finished cleaning express at $(date)"
