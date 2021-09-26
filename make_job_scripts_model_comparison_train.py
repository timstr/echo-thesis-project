import textwrap
import os
import sys

mode_cs = "cs"
mode_sockeye = "sockeye"


def print_usage_and_exit():
    print(f"usage: {sys.argv[0]} ({mode_cs} | {mode_sockeye})")
    exit(-1)


if len(sys.argv) != 2:
    print_usage_and_exit()

mode = sys.argv[1]

if mode not in [mode_cs, mode_sockeye]:
    print_usage_and_exit()


script_folder = "job_scripts"

if not os.path.exists(script_folder):
    os.makedirs(script_folder)


def make_script(model, dataset, batchsize, device):
    desc = f"model_{model}_{dataset}_{device.replace(':', '')}"

    if mode == mode_sockeye:
        preamble = f"""\
        #PBS -l walltime=72:00:00,select=1:ncpus=4:ngpus=1:mem=16gb
        #PBS -N {desc}
        #PBS -A st-rhodin-1-gpu
        #PBS -m abe
        #PBS -M timstr@cs.ubc.ca
        #PBS -o /scratch/st-rhodin-1/users/timstr/echo/logs/{desc}_OUTPUT.txt
        #PBS -e /scratch/st-rhodin-1/users/timstr/echo/logs/{desc}_ERROR.txt

        source activate /home/timstr/echo_env
        cd /home/timstr/echo
        """
        dataset_dir = "/project/st-rhodin-1/users/timstr"
        output_dir = "/scratch/st-rhodin-1/users/timstr/echo"
    elif mode == mode_cs:
        preamble = f"""\
        cd /scratch-ssd/timstr/echo/
        source ./init.sh
        """
        dataset_dir = "/scratch-ssd/timstr/echo/datasets"
        output_dir = "/scratch-ssd/timstr/echo/datasets/logs_from_training"

    contents = f"""\
    #!/bin/bash

    {preamble}

    export WAVESIM_DATASET_TRAIN={dataset_dir}/dataset_7.5mm_{dataset}_train.h5
    export WAVESIM_DATASET_VALIDATION={dataset_dir}/dataset_7.5mm_{dataset}_val.h5
    export TRAINING_LOG_PATH={output_dir}/logs
    export TRAINING_MODEL_PATH={output_dir}/models
    export MPLCONFIGDIR={output_dir}/matplotlib_junk

    echo "Starting training..."
    python3 train_model.py \\
        --experiment={desc} \\
        --model={model} \\
        --batchsize={batchsize} \\
        --iterations=1000000 \\
        --tofcropsize=256 \\
        --samplesperexample=256 \\
        --plotinterval=1024 \\
        --validationinterval=4096 \\
        --receivercountx=1 \\
        --receivercounty=2 \\
        --receivercountz=2 \\
        --chirpf0=18000.0 \\
        --chirpf1=22000.0 \\
        --chirplen=0.001 \\
        --validationdownsampling=4 \\
        --learningrate=2e-4 \\
        --adam_beta1=0.5 \\
        --adam_beta2=0.999

    echo "Training completed."
    """
    contents = textwrap.dedent(contents)
    script_name = f"{desc}.sh"
    with open(os.path.join(script_folder, script_name), "w", newline="\n") as f:
        f.write(contents)


for model, batchsize, device in [
    ("tofnet", 128, "cuda:0"),
    ("batvision_waveform", 16, "cuda:0"),
    ("batvision_spectrogram", 16, "cuda:1"),
    ("batgnet", 8, "cuda:1"),
]:
    make_script(model=model, dataset="random-outer", batchsize=batchsize, device=device)
