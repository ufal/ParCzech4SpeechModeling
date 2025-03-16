#!/bin/bash
#SBATCH -J whisper-ft-debug
#SBATCH -D /lnet/express/work/people/stankov/parczech/slurm-logs/whisper_ft_debug
#SBATCH -o large.out
#SBATCH -e large.out
#SBATCH -p gpu-troja,gpu-ms
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G

# Number of GPUs
#SBATCH --gres=gpu:nvidia_h100:1


conda_path=/lnet/express/work/people/stankov/miniconda3/bin/conda
export PATH=$conda_path:$PATH
eval "$(conda shell.bash hook)"

nvidia-smi

conda activate whisperx

python /lnet/express/work/people/stankov/parczech/scripts/whisper_ft.py 