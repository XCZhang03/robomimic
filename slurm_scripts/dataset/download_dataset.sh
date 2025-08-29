#!/bin/bash
#SBATCH --job-name download_dataset
#SBATCH -c 1                # Number of cores (-c)
#SBATCH -t 0-10:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p test   # Partition to submit to
#SBATCH --mem=10g           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o status/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e status/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid





python -m robomimic.scripts.download_datasets --tasks=all