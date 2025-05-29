#!/bin/bash
#SBATCH --output=logs/%x.%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -p cpu-troja,cpu-ms
#SBATCH -q low
#SBATCH --mem=64G

set -e
CMD="$1 THREADS=$SLURM_CPUS_ON_NODE"
/usr/bin/time -f "%x\t%E real, %U user, %S sys, %M kB" $CMD