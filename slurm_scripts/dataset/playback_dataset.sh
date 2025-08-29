#!/bin/bash
#SBATCH --job-name playback_dataset
#SBATCH -c 1                # Number of cores (-c)
#SBATCH -t 2-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p shared   # Partition to submit to
#SBATCH --mem=10g           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o status/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e status/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

source /n/holylabs/ydu_lab/Lab/zhangxiangcheng/miniconda3/etc/profile.d/conda.sh
conda activate robomimic
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/xczhang/DiffRL/robomimic_dataset/robomimic/

python -m robomimic.scripts.playback_pose_datasets --dataset all --noise 0.5