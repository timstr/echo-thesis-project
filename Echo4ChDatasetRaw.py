import sys
import torch
import torchvision
import glob
import os
import PIL.Image as Image
from device_dict import DeviceDict


class Echo4ChDatasetRaw(torch.utils.data.Dataset):
    def __init__(self):
        super(Echo4ChDatasetRaw, self).__init__()

        self._rootpath = os.environ.get("ECHO4CH")

        if self._rootpath is None:
            raise Exception(
                "Please set the ECHO4CH environment variable to point to the ECHO4CH dataset root"
            )

        sys.stdout.write("Loading ECHO4CH...")
        allfiles = glob.glob(f"{self._rootpath}/T*.png")

        if len(allfiles) == 0:
            raise Exception(
                "The ECHO4CH environment variable points to a folder which contains no example files. Windows users: did you accidentally include quotes in the environment variable?"
            )

        feature_groups = {}

        for filename in map(os.path.basename, allfiles):
            terms = filename.split(".")[0].split("_")
            shape_and_position, angle, repeat = terms[:3]
            example_name = f"{shape_and_position}_{angle}_{repeat}"
            feature_name = "_".join(terms[3:])

            if example_name not in feature_groups:
                feature_groups[example_name] = {}

            feature_groups[example_name][feature_name] = filename

        self._examples = []

        required_features = [
            "FR",
            "FL",
            "FU",
            "FD",
            "TR",
            "TL",
            "TU",
            "TD",
            "Label",
            "Label_depthmap",
        ]

        for example_name, features in feature_groups.items():
            for rf in required_features:
                if rf not in features:
                    continue

            self._examples.append(example_name)

        self._to_tensor = torchvision.transforms.ToTensor()

        sys.stdout.write(" Done.\n")

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, idx):
        example = self._examples[idx]
        basename = f"{self._rootpath}/{example}_"
        spectrograms = []
        # NOTE:
        # F ("high Frequency resolution") == LW ("Long Window")
        # T ("high Time resolution") == SW ("Short Window")
        for name in ["FR", "FL", "FU", "FD", "TR", "TL", "TU", "TD"]:
            img = Image.open(f"{basename}{name}.png")
            assert img.size == (256, 256)
            assert img.mode == "L"
            tensor = self._to_tensor(img)
            tensor = tensor.squeeze(0)
            spectrograms.append(tensor)

        spectrograms = torch.stack(spectrograms, dim=0)

        assert spectrograms.shape == (8, 256, 256)
        assert spectrograms.dtype == torch.float32

        occupancy_unfolded = Image.open(f"{basename}Label.png")
        # NOTE: PIL uses (width, height) convention
        assert occupancy_unfolded.size == (64, 4096)
        assert occupancy_unfolded.mode == "L"

        occupancy_unfolded = self._to_tensor(occupancy_unfolded)
        # Torch uses (height,width) convention
        assert occupancy_unfolded.shape == (1, 4096, 64)
        assert occupancy_unfolded.dtype == torch.float32
        occupancy = occupancy_unfolded.reshape(64, 64, 64)

        assert occupancy.dtype == torch.float32

        depthmap = Image.open(f"{basename}Label_depthmap.png")
        assert depthmap.size == (64, 64)
        assert depthmap.mode == "L"
        depthmap = self._to_tensor(depthmap)
        assert depthmap.shape == (1, 64, 64)
        assert depthmap.dtype == torch.float32
        depthmap = depthmap.squeeze(0)

        # Convert scalar to categorical, assuming values are either 0 or 1
        # Channel 0: empty label
        # Channel 1: occupied label
        # heatmap = torch.stack((
        #     1.0 - heatmap,
        #     heatmap
        # ), dim=0)
        # assert heatmap.shape == (2, 64, 64, 64)

        return DeviceDict(
            {"spectrograms": spectrograms, "occupancy": occupancy, "depthmap": depthmap}
        )
