import textwrap
from config_constants import output_format_depthmap, output_format_heatmap


def make_script(format, implicit, variance, summary_statistics):
    options_str = f"{format}_{'implicit' if implicit else 'dense'}_{'variance' if variance else 'novariance'}_{'summarystatistics' if summary_statistics else 'nosummarystatistics'}"
    desc = f"echo4ch_latest_{options_str}"
    modelpath = f"D:\\ultrasonic\\sockeye\\models\\echo4ch_24h_no_timestamp\\latest\\train_echo4ch_{options_str}_latest.dat"
    outputpath = f"output_echo4ch_latest\\{desc}.txt"
    contents = f"""\
python echotest.py \
--description={desc} \
--dataset=echo4ch \
--receivercount=8 \
--receiverarrangement=grid \
--emitterarrangement=mono \
--emittersignal=sweep \
--emittersequential=false \
--emittersamefrequency=false \
--implicitfunction={implicit} \
--predictvariance={variance} \
--resolution=64 \
--nninput=spectrogram \
--nnoutput={format} \
--summarystatistics={summary_statistics} \
--modelpath={modelpath} \
--makeimages \
--computemetrics \
--occupancyshadows \
> {outputpath}
    """
    contents = "".join(textwrap.dedent(contents).split("\n"))

    print(contents)
    print("\n\n")


print("\n")

for format in [output_format_depthmap, output_format_heatmap]:
    for implicit in [False, True]:
        for variance in [False, True]:
            for summary_statistics in [False, True]:
                make_script(format, implicit, variance, summary_statistics)
