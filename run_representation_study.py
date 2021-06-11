import subprocess
from config_constants import (
    output_format_sdf,
    output_format_heatmap,
    output_format_depthmap,
)


def run_process(representation, implicitfunction):
    arg_repr = "--representation={}".format(representation)
    arg_imfn = "--implicitfunction" if implicitfunction else ""
    arg_xpnm = (
        "--experiment=representation_study_biggercnn_"
        + representation
        + ("_implicitfn" if implicitfunction else "_denseoutput")
    )

    args_all = " ".join([arg_xpnm, arg_repr, arg_imfn])
    cmd_base = "python echolearn.py --experiment=representation_study --dataset=v8 --batchsize=16 --receivers=8 --arrangement=grid --maxobstacles=2 --iterations=60000"
    cmd_full = cmd_base + " " + args_all
    subprocess.run(cmd_full)


def main():
    for representation in [
        output_format_sdf,
        output_format_heatmap,
        output_format_depthmap,
    ]:
        for implicitfunction in [False, True]:
            run_process(representation, implicitfunction)


if __name__ == "__main__":
    main()
