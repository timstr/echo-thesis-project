import scipy.io.wavfile as wf
import matplotlib.pyplot as plt
import glob

from featurize_audio import make_spectrogram

def main():
    filenames = sorted(glob.glob("*.wav"))

    for fn in filenames:
        if fn.find("waveshaped") > 0:
            continue
        sfreq, data = wf.read(fn)
        
        spectrogram = make_spectrogram(data)

        plt.imshow(spectrogram)
        plt.gcf().suptitle(fn)
        plt.show()

if __name__ == "__main__":
    main()
