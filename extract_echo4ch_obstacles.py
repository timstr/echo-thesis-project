import os
import sys
import h5py
import glob
import numpy as np
import PIL.Image as Image

from h5ds import H5DS
from tof_utils import progress_bar


def main():
    rootpath = os.environ.get("ECHO4CH")

    if rootpath is None:
        raise Exception(
            "Please set the ECHO4CH environment variable to point to the ECHO4CH dataset root"
        )

    sys.stdout.write("Loading ECHO4CH...")
    sys.stdout.flush()

    digit = "[0-9]"

    allfiles = sorted(
        glob.glob(os.path.join(rootpath, f"T{digit * 6}_{digit * 2}_01_Label.png"))
        # glob.glob(os.path.join(rootpath, f"T{digit * 6}_01_01_Label.png"))
    )

    sys.stdout.write(" done.\n")
    sys.stdout.flush()

    if len(allfiles) == 0:
        raise Exception(
            "The ECHO4CH environment variable points to a folder which contains no example files. Windows users: did you accidentally include quotes in the environment variable?"
        )

    print(f"{len(allfiles)} distinct obstacles were found")

    path_to_h5file = "echo4ch_obstacles.h5"

    # IMPORTANT: don't open the file in truncate mode or the dataset will be gone
    h5file = h5py.File(path_to_h5file, "a")

    obstacle_ds = H5DS(
        name="obstacles", dtype=np.bool8, shape=(64, 64, 64), extensible=True
    )

    assert not obstacle_ds.exists(h5file)

    obstacle_ds.create(h5file)

    for i_file, file in enumerate(allfiles):
        progress_bar(i_file, len(allfiles))
        occupancy_unfolded = Image.open(os.path.join(rootpath, file))
        # NOTE: PIL uses (width, height) convention
        assert occupancy_unfolded.size == (64, 4096)
        assert occupancy_unfolded.mode == "L"
        occupancy_unfolded = np.array(occupancy_unfolded)

        occupancy = occupancy_unfolded.reshape(64, 64, 64)
        assert np.min(occupancy) == 0
        assert np.max(occupancy) == 255
        occupancy = occupancy > 127
        obstacle_ds.append(h5file, occupancy)


if __name__ == "__main__":
    main()
