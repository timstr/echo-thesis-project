import textwrap
import os

out_folder = "job_scripts"

if not os.path.exists(out_folder):
    os.makedirs(out_folder)


def make_script(model, training_dataset, test_dataset, model_path, backfill, hours):
    desc = f"eval_{model}__train_{training_dataset}__test_{test_dataset}_{'backfill' if backfill else 'nobackfill'}"
    contents = f"""\
    #!/bin/bash

    #PBS -l walltime={hours}:00:00,select=1:ncpus=4:ngpus=1:mem=16gb
    #PBS -N {desc}
    #PBS -A st-rhodin-1-gpu
    #PBS -m abe
    #PBS -M timstr@cs.ubc.ca
    #PBS -o /scratch/st-rhodin-1/users/timstr/echo/logs/{desc}_OUTPUT.txt
    #PBS -e /scratch/st-rhodin-1/users/timstr/echo/logs/{desc}_ERROR.txt

    source activate /home/timstr/echo_env
    cd /home/timstr/echo

    export WAVESIM_DATASET_VALIDATION=/project/st-rhodin-1/users/timstr/dataset_7.5mm_{training_dataset}_val.h5
    export WAVESIM_DATASET_TEST=/project/st-rhodin-1/users/timstr/dataset_7.5mm_{test_dataset}_test.h5
    export TRAINING_LOG_PATH=/scratch/st-rhodin-1/users/timstr/echo/logs
    export TRAINING_MODEL_PATH=/scratch/st-rhodin-1/users/timstr/echo/models
    export MPLCONFIGDIR=/scratch/st-rhodin-1/users/timstr/matplotlib_junk

    echo "Starting evaluation..."
    python3 evaluate_model.py \\
        --experiment={desc} \\
        --model={model} \\
        --tofcropsize=128 \\
        --receivercountx=1 \\
        --receivercounty=2 \\
        --receivercountz=2 \\
        --chirpf0=18000.0 \\
        --chirpf1=22000.0 \\
        --chirplen=0.001 \\
        --restoremodelpath={model_path} \\
        {'--backfill' if backfill else ''} \\
        --offsetsdf

    echo "Evaluation done."
    """
    contents = textwrap.dedent(contents)
    script_name = f"{desc}.sh"
    with open(os.path.join(out_folder, script_name), "w", newline="\n") as f:
        f.write(contents)


bgn = "batgnet"
bvw = "batvision_waveform"
bvs = "batvision_spectrogram"
tof = "tofnet"

dsr = "random"
dsi = "random-inner"
dso = "random-outer"

models_paths_and_whether_to_backfill = [
    (bgn, dsr, "model_batgnet_random_16-09-2021_10-44-01", [False]),
    (bgn, dsi, "model_batgnet_random-inner_16-09-2021_10-43-42", [False]),
    (bgn, dso, "model_batgnet_random-outer_18-09-2021_15-12-21", [False]),
    (bvs, dsr, "model_batvision_spectrogram_random_16-09-2021_10-44-23", [True]),
    (bvs, dsi, "model_batvision_spectrogram_random-inner_16-09-2021_10-44-15", [True]),
    (bvs, dso, "model_batvision_spectrogram_random-outer_18-09-2021_15-14-08", [True]),
    (bvw, dsr, "model_batvision_waveform_random_16-09-2021_10-45-05", [True]),
    (bvw, dsi, "model_batvision_waveform_random-inner_16-09-2021_10-44-55", [True]),
    (bvw, dso, "model_batvision_waveform_random-outer_18-09-2021_15-14-12", [True]),
    (tof, dsr, "model_tofnet_random_16-09-2021_10-45-45", [False, True]),
    (tof, dsi, "model_tofnet_random-inner_16-09-2021_10-45-16", [False, True]),
    (tof, dso, "model_tofnet_random-outer_18-09-2021_15-14-26", [False, True]),
]

train_to_test_datasets = {dsr: [dsr], dsi: [dsi, dso], dso: [dsi, dso]}

model_to_hours = {bgn: 1, bvw: 1, bvs: 1, tof: 6}

for (
    model,
    training_dataset,
    model_folder_name,
    whether_to_backfill,
) in models_paths_and_whether_to_backfill:
    for test_dataset in train_to_test_datasets[training_dataset]:
        for backfill in whether_to_backfill:
            make_script(
                model=model,
                training_dataset=training_dataset,
                test_dataset=test_dataset,
                model_path=f"/scratch/st-rhodin-1/users/timstr/echo/models/{model_folder_name}/model_{model}_best.dat",
                backfill=backfill,
                hours=model_to_hours[model],
            )
