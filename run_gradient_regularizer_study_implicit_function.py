import subprocess


def run_process(regularizerValue):
    arg_rglr = (
        "--gradientregularizer=" + str(regularizerValue)
        if (regularizerValue is not None)
        else ""
    )
    arg_xpnm = "--experiment=gradient_regularizer_study_implicitfn" + (
        "_regularizer_" + str(regularizerValue)
        if (regularizerValue is not None)
        else "_noregularizer"
    )

    args_all = " ".join([arg_xpnm, arg_rglr])
    cmd_base = "python echolearn.py --dataset=v8 --batchsize=4 --receivers=8 --arrangement=grid --iterations=50000 --nninput=spectrogram --nnoutput=sdf --implicitfunction"
    cmd_full = cmd_base + " " + args_all
    subprocess.run(cmd_full)


def main():
    for regularizerValue in [
        None,
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
        1000.0,
        10000.0,
        100000.0,
        1000000.0,
    ]:
        run_process(regularizerValue)


if __name__ == "__main__":
    main()
