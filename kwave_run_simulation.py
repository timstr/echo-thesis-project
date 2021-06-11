from kwave_simulation_description import SimulationDescription
import subprocess
import os
import h5py
import numpy as np
import scipy.interpolate as interpolate
import scipy.signal as signal


def run_kwave_simulation(description):
    assert isinstance(description, SimulationDescription)

    kwave_executable = os.environ.get("KWAVE_EXECUTABLE")

    if kwave_executable is None or not os.path.isfile(kwave_executable):
        raise Exception(
            "Please set the KWAVE_EXECUTABLE environment variable to point to a k-wave executable"
        )

    hdf5_input_file_path = "temp_kwave_simulation_input.h5"
    hdf5_output_file_path = "temp_kwave_simulation_output.h5"

    description.write(hdf5_input_file_path)

    subprocess.run(
        [kwave_executable, "-i", hdf5_input_file_path, "-o", hdf5_output_file_path],
    )

    with h5py.File(hdf5_output_file_path, "r") as interpolation_function:
        pressure_vs_time = np.array(interpolation_function["p"])
        assert pressure_vs_time.shape == (
            1,
            description.Nt,
            description.num_sensors,
        )

    signals = pressure_vs_time[0].transpose(1, 0)
    assert signals.shape == (description.num_sensors, description.Nt)

    print(
        f"Low-passing the simulated signal at {description.output_sampling_frequency} Hz before resampling"
    )
    sos = signal.butter(
        N=10,
        Wn=description.output_sampling_frequency,
        fs=description.simulation_sampling_frequency,
        btype="lowpass",
        output="sos",
    )

    print(
        f"Resampling the low-passed signal to {description.output_sampling_frequency} Hz"
    )
    signals_lowpassed = signal.sosfilt(sos=sos, x=signals, axis=1)

    assert signals_lowpassed.shape == (
        description.num_sensors,
        description.Nt,
    )

    # This introduces high-frequency artefacts which I do not like one bit
    # signals_resampled = signal.resample(
    #     signals_lowpassed, num=description.output_length, axis=1
    # )

    x_old = np.linspace(0, 1.0, num=description.Nt, endpoint=True)
    interpolation_function = interpolate.interp1d(
        x=x_old, y=signals, kind="linear", axis=1
    )
    x_new = np.linspace(0.0, 1.0, num=description.output_length, endpoint=True)

    signals_resampled = interpolation_function(x_new)

    assert signals_resampled.shape == (
        description.num_sensors,
        description.output_length,
    )

    os.remove(hdf5_input_file_path)
    os.remove(hdf5_output_file_path)

    return signals_resampled
