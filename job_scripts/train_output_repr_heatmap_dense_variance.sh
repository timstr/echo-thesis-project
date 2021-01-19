#!/bin/bash

#PBS -l walltime=18:00:00,select=1:ncpus=4:ngpus=1:mem=16gb
#PBS -N train_output_repr_heatmap_dense_variance
#PBS -A st-rhodin-1-gpu
#PBS -m abe
#PBS -M timstr@cs.ubc.ca
#PBS -o /scratch/st-rhodin-1/users/timstr/echo/logs/train_output_repr_heatmap_dense_variance_OUTPUT.txt
#PBS -e /scratch/st-rhodin-1/users/timstr/echo/logs/train_output_repr_heatmap_dense_variance_ERROR.txt

source activate /home/timstr/echo_env
cd /home/timstr/echo

export WAVESIM_DATASET=/project/st-rhodin-1/users/timstr/wavesim_dataset
export TRAINING_LOG_PATH=/scratch/st-rhodin-1/users/timstr/echo/logs
export TRAINING_MODEL_PATH=/scratch/st-rhodin-1/users/timstr/echo/models

echo "Starting training..."
python3 echolearn.py \
    --experiment=train_output_repr_heatmap_dense_variance \
    --batchsize=64 \
    --receivercount=8 \
    --receiverarrangement=grid \
    --emitterarrangement=mono \
    --emittersignal=sweep \
    --allowocclusions \
    --nodisplay \
     \
    --importancesampling \
    --predictvariance \
    --resolution=256 \
    --nninput=spectrogram \
    --nnoutput=heatmap \
    --summarystatistics \
    --plotinterval=64 \
    --validationinterval=1024

echo "Training completed."
