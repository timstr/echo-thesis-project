import compress_pickle
from progress_bar import progress_bar
from device_dict import DeviceDict
import os
import torch
import math

from Echo4ChDatasetRaw import Echo4ChDatasetRaw


def convert_example(d):
    assert isinstance(d, DeviceDict)

    spectrograms = d["spectrograms"]
    occupancy = d["occupancy"]
    depthmap = d["depthmap"]
    assert isinstance(spectrograms, torch.FloatTensor)
    assert isinstance(occupancy, torch.FloatTensor)
    assert isinstance(depthmap, torch.FloatTensor)
    assert spectrograms.shape == (8, 256, 256)
    assert occupancy.shape == (64, 64, 64)
    assert depthmap.shape == (64, 64)

    spectrograms *= 255.0
    occupancy *= 255.0
    depthmap *= 255.0

    assert torch.all(spectrograms >= 0.0)
    assert torch.all(spectrograms <= 255.0)
    assert torch.all(occupancy >= 0.0)
    assert torch.all(occupancy <= 255.0)
    assert torch.all(depthmap >= 0.0)
    assert torch.all(depthmap <= 255.0)
    out_s = spectrograms.to(torch.uint8).numpy()
    out_o = occupancy.to(torch.uint8).numpy()
    out_d = depthmap.to(torch.uint8).numpy()

    return {"spectrograms": out_s, "occupancy": out_o, "depthmap": out_d}


def main():
    e4c = Echo4ChDatasetRaw()

    output_path = os.environ.get("NEW_DATASET_PATH")

    if output_path is None or not os.path.exists(output_path):
        raise Exception(
            "Please set the NEW_DATASET_PATH environment variable to point to the desired new dataset directory"
        )

    dataset_size = len(e4c)

    num_digits = int(math.ceil(math.log10(dataset_size)))

    for i, example in enumerate(e4c):
        converted = convert_example(example)
        fname = f"example {str(i).zfill(num_digits)}.pkl"
        path = os.path.join(output_path, fname)
        with open(path, "wb") as outfile:
            compress_pickle.dump(converted, outfile, "gzip")
        progress_bar(i, dataset_size)


if __name__ == "__main__":
    main()
