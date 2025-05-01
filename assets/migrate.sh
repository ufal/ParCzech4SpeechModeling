#!/bin/bash
#SBATCH --job-name=rsync_21
#SBATCH -D /lnet/work/people/stankov/parczech/slurm-logs/migrate
#SBATCH -o rsync-21.log
#SBATCH -e rsync-21.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G


srcdir="/lnet/express/work/people/stankov/"
destdir="/lnet/troja/work/people/stankov/backup"

echo "Removing full from the destination at $(date)"
rm -rf /lnet/troja/work/people/stankov/backup/alignment/results/full

echo "Removing full from the source at $(date)"
rm -rf /lnet/express/work/people/stankov/alignment/results/full

echo "Removing done at $(date)"

echo "Starting rsync from $srcdir to $destdir at $(date)"

mkdir -p $destdir

rsync -a --info=progress2 --inplace --rsync-path="ionice -c1 rsync" --bwlimit=0  --size-only --ignore-existing "$srcdir" "$destdir"

echo "Finished rsync for $destdir at $(date)"
