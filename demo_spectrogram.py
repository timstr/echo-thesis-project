import scipy.io.wavfile as wf
import matplotlib.pyplot as plt
import glob

from featurize_audio import make_spectrogram

def main():
    filenames = sorted(glob.glob("*simulation*.wav"))

    for fn in filenames:
        if fn.find("waveshaped") > 0:
            continue
        sfreq, data = wf.read(fn)
        
        try:
            spectrogram = make_spectrogram(data)
        except:
            print(f"I don't like this file: {fn}")
            continue

        plt.imshow(spectrogram)
        plt.gcf().suptitle(fn)
        plt.show()

if __name__ == "__main__":
    main()
