#!/bin/bash
#SBATCH -J whisper-ft-debug
#SBATCH -D /lnet/work/people/stankov/parczech/slurm-logs/whisper_ft_debug
#SBATCH -o large.out
#SBATCH -e large.out
#SBATCH -p gpu-troja,gpu-ms
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G

# Number of GPUs
#SBATCH --gres=gpu:nvidia_h100:1

export HF_HOME=/lnet/work/people/stankov/huggingface_cache
conda_path=/lnet/work/people/stankov/miniconda3/bin/conda
export PATH=$conda_path:$PATH
eval "$(conda shell.bash hook)"

nvidia-smi

conda activate whisperx

python /lnet/work/people/stankov/parczech/scripts/whisper_ft.py 