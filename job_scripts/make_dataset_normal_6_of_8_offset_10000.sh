#!/bin/bash

#PBS -l walltime=2:00:00,select=1:ncpus=4:ngpus=1:mem=16gb
#PBS -N make_dataset_normal_6_of_8_offset_10000
#PBS -A st-rhodin-1-gpu
#PBS -m abe
#PBS -M timstr@cs.ubc.ca
#PBS -o /scratch/st-rhodin-1/users/timstr/echo/logs/make_dataset_normal_6_of_8_offset_10000_OUTPUT.txt
#PBS -e /scratch/st-rhodin-1/users/timstr/echo/logs/make_dataset_normal_6_of_8_offset_10000_ERROR.txt

source activate /home/timstr/echo_env
cd /home/timstr/echo

export OUTPUT_PATH=/scratch/st-rhodin-1/users/timstr/echo/output_dataset_normal

echo "Starting dataset creation..."
python3 make_dataset.py \
    --offset=10000 \
    --end=11000 \
    --numworkers=8 \
    --workerindex=5 \
    --mode=normal

echo "Dataset creation completed."
