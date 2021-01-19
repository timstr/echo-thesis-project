#!/bin/bash

#PBS -l walltime=12:00:00,select=1:ncpus=4:ngpus=1:mem=16gb
#PBS -N make_dataset_orbiting_circle_2_of_4
#PBS -A st-rhodin-1-gpu
#PBS -m abe
#PBS -M timstr@cs.ubc.ca
#PBS -o /scratch/st-rhodin-1/users/timstr/echo/logs/make_dataset_orbiting_circle_2_of_4_OUTPUT.txt
#PBS -e /scratch/st-rhodin-1/users/timstr/echo/logs/make_dataset_orbiting_circle_2_of_4_ERROR.txt

source activate /home/timstr/echo_env
cd /home/timstr/echo

export OUTPUT_PATH=/scratch/st-rhodin-1/users/timstr/echo/output_dataset_orbiting_circle

echo "Starting dataset creation..."
python3 make_dataset.py \
    --offset=0 \
    --end=10000 \
    --numworkers=4 \
    --workerindex=1 \
    --mode=orbiting-circle

echo "Dataset creation completed."
