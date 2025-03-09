#!/bin/bash
#SBATCH -J parcz-debug
#SBATCH -D /lnet/express/work/people/stankov/parczech/slurm-logs
#SBATCH -o parcz-debug.out
#SBATCH -e parcz-debug.err
#SBATCH -p gpu-troja,gpu-ms
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

# Number of GPUs
#SBATCH --gres=gpu:nvidia_a30:1


id=debug
links="[https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-5404/audioPSP-2013-Q4.tar,https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-5404/audioPSP-2017-Q4.tar]"

conda_path=/lnet/express/work/people/stankov/miniconda3/bin/conda
export PATH=$conda_path:$PATH
eval "$(conda shell.bash hook)"

conda activate whisperx

# Your job command(s)
echo "Running on node: $(hostname)"
echo "Job started at: $(date)"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi

python /lnet/express/work/people/stankov/parczech/scripts/recognize.py \
links=$links overwrite_recognized=True
