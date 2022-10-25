import fix_dead_command_line
import cleanup_when_killed

import os
import torch
import torch.nn.functional as F
import math
from argparse import ArgumentParser
import PIL

from batgnet import BatGNet
from Batvision.Models import (
    WaveformNet as BatVisionWaveform,
    SpectrogramNet as BatVisionSpectrogram,
)

from network_utils import split_network_prediction
from visualization import (
    raymarch_sdf_ground_truth,
    raymarch_sdf_prediction,
    vector_cross,
    vector_length,
    vector_normalize,
)
from signals_and_geometry import (
    backfill_depthmap,
    backfill_occupancy,
    make_fm_chirp,
    obstacle_map_to_sdf,
)
from split_till_it_fits import SplitSize, split_till_it_fits
from assert_eq import assert_eq
from which_device import get_compute_device
from current_simulation_description import (
    all_grid_locations,
    make_receiver_indices,
    make_simulation_description,
    minimum_x_units,
)
from dataset3d import WaveDataset3d, k_sensor_recordings, k_sdf, k_obstacles
from dataset_adapters import (
    convolve_recordings_dict,
    sclog_dict,
    subset_recordings_dict,
    wavesim_to_batgnet_spectrogram,
    wavesim_to_batvision_spectrogram,
    wavesim_to_batvision_waveform,
)
from time_of_flight_net import TimeOfFlightNet
from torch_utils import restore_module


model_tof_net = "tofnet"
model_batvision_waveform = "batvision_waveform"
model_batvision_spectrogram = "batvision_spectrogram"
model_batgnet = "batgnet"

view_perspective = "perspective"
view_front = "front"
view_top = "top"
view_side = "side"
view_front_perspective = "front_perspective"


def main():
    parser = ArgumentParser()
    parser.add_argument("path_to_dataset", type=str)
    parser.add_argument("indices", nargs="*", metavar="indices", type=int, default=0)
    parser.add_argument(
        "--tofcropsize",
        type=int,
        dest="tofcropsize",
        default=256,
        help="Number of samples used in time-of-flight crop",
    )
    parser.add_argument(
        "--chirpf0",
        type=float,
        dest="chirpf0",
        default=18000.0,
        help="chirp start frequency (Hz)",
    )
    parser.add_argument(
        "--chirpf1",
        type=float,
        dest="chirpf1",
        default=22000.0,
        help="chirp end frequency (Hz)",
    )
    parser.add_argument(
        "--chirplen",
        type=float,
        dest="chirplen",
        default=0.001,
        help="chirp duration (seconds)",
    )
    parser.add_argument("--receivercountx", type=int, dest="receivercountx", default=1)
    parser.add_argument("--receivercounty", type=int, dest="receivercounty", default=2)
    parser.add_argument("--receivercountz", type=int, dest="receivercountz", default=2)
    parser.add_argument(
        "--restoremodelpath", type=str, dest="restoremodelpath", default=None
    )
    parser.add_argument("--outputpath", type=str, dest="outputpath", required=True)
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            model_tof_net,
            model_batvision_waveform,
            model_batvision_spectrogram,
            model_batgnet,
        ],
        dest="model",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--precompute",
        dest="precompute",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--showsensor",
        dest="showsensor",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--show_sdf_plane",
        dest="show_sdf_plane",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--show_axes",
        dest="show_axes",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--backfill",
        dest="backfill",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--sclog",
        dest="sclog",
        default=False,
        action="store_true",
    )
    parser.add_argument("--supersampling", type=int, dest="supersampling", default=None)
    parser.add_argument(
        "--view",
        type=str,
        choices=[
            view_perspective,
            view_front,
            view_side,
            view_top,
            view_front_perspective,
        ],
        dest="view",
        required=False,
        default=view_perspective,
    )
    args = parser.parse_args()

    apply_sclog = args.sclog
    which_model = args.model
    if args.restoremodelpath is not None and which_model is None:
        for model_type in [
            model_tof_net,
            model_batvision_waveform,
            model_batvision_spectrogram,
            model_batgnet,
        ]:
            if model_type in args.restoremodelpath:
                which_model = model_type
                print(
                    f'NOTE: the model type "{which_model}" was inferred from the model path'
                )
        if which_model is None:
            raise Exception(
                f"ERROR: please specify a model type for the model path {args.restoremodelpath}"
            )
        if "_scl_" in args.restoremodelpath and not apply_sclog:
            apply_sclog = True
            print(
                f'NOTE: apply sclog because "_scl_" was found in the model path {args.restoremodelpath}'
            )

    if args.restoremodelpath is None:
        prediction = False
    else:
        prediction = True
        assert args.tofcropsize is not None
        assert args.chirpf0 is not None
        assert args.chirpf1 is not None
        assert args.chirplen is not None
        assert args.precompute is not None
        assert args.receivercountx is not None
        assert args.receivercounty is not None
        assert args.receivercountz is not None

    if args.showsensor:
        assert args.receivercountx is not None
        assert args.receivercounty is not None
        assert args.receivercountz is not None

    if args.supersampling is None:
        supersampling = 1
    else:
        supersampling = args.supersampling
        assert supersampling >= 1

    description = make_simulation_description()

    dataset = WaveDataset3d(description, args.path_to_dataset)

    raymarch_split_size = SplitSize("raymarch")

    network_prediction_split_size = SplitSize("network_prediction")

    if prediction and (which_model != model_tof_net):
        assert_eq(args.receivercountx, 1)
        assert_eq(args.receivercounty, 2)
        assert_eq(args.receivercountz, 2)

    sensor_indices = make_receiver_indices(
        args.receivercountx,
        args.receivercounty,
        args.receivercountz,
    )

    if args.showsensor:
        raymarch_emitter_location = torch.tensor(description.emitter_location).to(
            get_compute_device()
        )
        assert_eq(raymarch_emitter_location.shape, (3,))
        raymarch_receiver_locations = (
            torch.tensor(description.sensor_locations[sensor_indices])
            .permute(1, 0)
            .to(get_compute_device())
        )
        assert_eq(raymarch_receiver_locations.shape, (3, len(sensor_indices)))
    else:
        raymarch_emitter_location = None
        raymarch_receiver_locations = None

    if prediction:
        fm_chirp = make_fm_chirp(
            begin_frequency_Hz=args.chirpf0,
            end_frequency_Hz=args.chirpf1,
            sampling_frequency=description.output_sampling_frequency,
            chirp_length_samples=math.ceil(
                args.chirplen * description.output_sampling_frequency
            ),
            wave="sine",
            device=get_compute_device(),
        )

        def adapt_signals(dd):
            dd = convolve_recordings_dict(
                subset_recordings_dict(dd, sensor_indices), fm_chirp
            )
            if apply_sclog:
                dd = sclog_dict(dd)
            return dd

        if which_model == model_tof_net:
            the_model = TimeOfFlightNet(
                speed_of_sound=description.air_properties.speed_of_sound,
                sampling_frequency=description.output_sampling_frequency,
                recording_length_samples=description.output_length,
                crop_length_samples=args.tofcropsize,
                emitter_location=description.emitter_location,
                receiver_locations=description.sensor_locations[sensor_indices],
                use_convolutions=True,
                use_fourier_transform=False,
                kernel_size=31,
                hidden_features=128,
                no_amplitude_compensation=False,
            )
        elif which_model == model_batvision_waveform:
            the_model = BatVisionWaveform(generator="direct")
        elif which_model == model_batvision_spectrogram:
            the_model = BatVisionSpectrogram(generator="unet")
        elif which_model == model_batgnet:
            the_model = BatGNet()

        restore_module(the_model, args.restoremodelpath)
        the_model = the_model.to(get_compute_device())
        the_model.eval()

    for index in args.indices:
        print(f"Rendering example {index}")
        example = dataset[index].to(get_compute_device())
        if prediction:
            example = adapt_signals(example)

        if args.view == view_top:
            # top view
            rm_camera_center = [
                0.5 * (description.xmin + description.xmax),
                description.ymin - 0.01,
                0.5 * (description.zmin + description.zmax),
            ]
            rm_camera_up = [0.0, 0.0, 0.55 * description.Nx * description.dx]
            rm_camera_right = [0.55 * description.Nx * description.dx, 0.0, 0.0]
            rm_x_resolution = 1024
            rm_y_resolution = 1024
            rm_fov_deg = 0.0
        elif args.view == view_side:
            # side view
            rm_camera_center = [
                0.5 * (description.xmin + description.xmax),
                0.5 * (description.ymin + description.ymax),
                description.zmax + 0.01,
            ]
            rm_camera_up = [0.0, 0.55 * description.Nx * description.dx, 0.0]
            rm_camera_right = [0.55 * description.Nx * description.dx, 0.0, 0.0]
            rm_x_resolution = 1024
            rm_y_resolution = 1024
            rm_fov_deg = 0.0
        elif args.view in [view_front, view_front_perspective]:
            # front view
            rm_camera_center = [description.xmin - 0.01, 0.0, 0.0]
            rm_camera_up = [0.0, 0.55 * description.Nx * description.dx, 0.0]
            rm_camera_right = [0.0, 0.0, 0.55 * description.Nx * description.dx]
            rm_x_resolution = 1024
            rm_y_resolution = 1024
            rm_fov_deg = 0.0 if args.view == view_front else 10.0

        elif args.view == view_perspective:
            # perspective view
            rm_camera_center = [-0.445, -0.4, 1.0]
            rm_camera_up = vector_normalize([-0.2, 1.0, 0.2], norm=0.6)
            rm_camera_right = vector_normalize([1.0, 0.0, 1.0], norm=1.2)
            rm_x_resolution = 1024 * 2
            rm_y_resolution = 512 * 2
            rm_fov_deg = 10.0

        # Make sure that camera directions are orthogonal
        assert (
            abs(
                1.0
                - vector_length(
                    vector_cross(
                        vector_normalize(rm_camera_up),
                        vector_normalize(rm_camera_right),
                    )
                )
            )
            < 1e-6
        )

        if prediction and which_model == model_tof_net:

            if args.precompute and which_model == model_tof_net:
                locations = all_grid_locations(
                    get_compute_device(), description, downsample_factor=1
                )

                x_steps = description.Nx - minimum_x_units
                y_steps = description.Ny
                z_steps = description.Nz

                offset = -0.5 * (supersampling - 1) / supersampling

                sdf_pred_acc = torch.zeros(
                    (x_steps * y_steps * z_steps,), device=get_compute_device()
                )

                for ss in range(supersampling ** 3):
                    ss_i = ss // supersampling ** 2
                    ss_j = ss // supersampling % supersampling
                    ss_k = ss % supersampling

                    ss_dx = description.dx * (offset + (ss_i / supersampling))
                    ss_dy = description.dy * (offset + (ss_j / supersampling))
                    ss_dz = description.dz * (offset + (ss_k / supersampling))

                    locations_offset = locations + torch.tensor(
                        [[ss_dx, ss_dy, ss_dz]], device=get_compute_device()
                    )

                    sdf_pred_acc += split_till_it_fits(
                        split_network_prediction,
                        network_prediction_split_size,
                        model=the_model,
                        locations=locations_offset,
                        recordings=example[k_sensor_recordings],
                        description=description,
                        show_progress_bar=True,
                    )

                sdf_pred = sdf_pred_acc / (supersampling ** 3)
                sdf_pred = sdf_pred.reshape(x_steps, y_steps, z_steps)

                obstacle_map = torch.zeros(
                    (description.Nx, description.Ny, description.Nz),
                    device=get_compute_device(),
                    dtype=torch.bool,
                )
                obstacle_map[minimum_x_units:] = sdf_pred <= 0.0

                if args.backfill:
                    obstacle_map = backfill_occupancy(obstacle_map)

                obstacle_sdf = obstacle_map_to_sdf(obstacle_map, description)

                img = split_till_it_fits(
                    raymarch_sdf_ground_truth,
                    raymarch_split_size,
                    camera_center_xyz=rm_camera_center,
                    camera_up_xyz=rm_camera_up,
                    camera_right_xyz=rm_camera_right,
                    x_resolution=rm_x_resolution,
                    y_resolution=rm_y_resolution,
                    description=description,
                    obstacle_sdf=obstacle_sdf,
                    emitter_location=raymarch_emitter_location,
                    receiver_locations=raymarch_receiver_locations,
                    field_of_view_degrees=rm_fov_deg,
                    show_sdf_plane=args.show_sdf_plane,
                    show_axes=args.show_axes,
                )
            else:
                img = split_till_it_fits(
                    raymarch_sdf_prediction,
                    split_size=raymarch_split_size,
                    camera_center_xyz=rm_camera_center,
                    camera_up_xyz=rm_camera_up,
                    camera_right_xyz=rm_camera_right,
                    x_resolution=rm_x_resolution,
                    y_resolution=rm_y_resolution,
                    description=description,
                    model=the_model,
                    recordings=example[k_sensor_recordings],
                    emitter_location=raymarch_emitter_location,
                    receiver_locations=raymarch_receiver_locations,
                    field_of_view_degrees=rm_fov_deg,
                    show_sdf_plane=args.show_sdf_plane,
                    show_axes=args.show_axes,
                )
        else:
            if prediction:
                obstacle_map_pred = torch.zeros(
                    (description.Nx, description.Ny, description.Nz),
                    device=get_compute_device(),
                    dtype=torch.bool,
                )
                if which_model == model_batgnet:
                    inputs = wavesim_to_batgnet_spectrogram(example)
                    occupancy_pred = the_model(inputs.unsqueeze(0)).squeeze(0)
                    occupancy_pred_resampled = (
                        F.interpolate(
                            occupancy_pred.unsqueeze(0).unsqueeze(0),
                            size=(
                                description.Nx - minimum_x_units,
                                description.Ny,
                                description.Nz,
                            ),
                            mode="trilinear",
                            align_corners=False,
                        )
                        .squeeze(0)
                        .squeeze(0)
                        > 0.5
                    )
                    obstacle_map_pred[minimum_x_units:].copy_(occupancy_pred_resampled)
                elif which_model in [
                    model_batvision_waveform,
                    model_batvision_spectrogram,
                ]:
                    if which_model == model_batvision_waveform:
                        inputs = wavesim_to_batvision_waveform(example)
                    else:
                        inputs = wavesim_to_batvision_spectrogram(example)
                    depthmap_pred = the_model(inputs.unsqueeze(0)).squeeze(0)
                    occupancy_pred = backfill_depthmap(
                        depthmap_pred, Nx=description.Nx - minimum_x_units
                    ).float()
                    occupancy_pred_resampled = (
                        F.interpolate(
                            occupancy_pred.unsqueeze(0).unsqueeze(0),
                            size=(
                                description.Nx - minimum_x_units,
                                description.Ny,
                                description.Nz,
                            ),
                            mode="trilinear",
                            align_corners=False,
                        )
                        .squeeze(0)
                        .squeeze(0)
                        > 0.5
                    )
                    obstacle_map_pred[minimum_x_units:].copy_(occupancy_pred_resampled)
                obstacle_sdf = obstacle_map_to_sdf(obstacle_map_pred, description)
            else:
                if args.backfill:
                    obstacle_sdf = obstacle_map_to_sdf(
                        backfill_occupancy(example[k_obstacles]), description
                    )
                else:
                    obstacle_sdf = example[k_sdf]

            img = split_till_it_fits(
                raymarch_sdf_ground_truth,
                raymarch_split_size,
                camera_center_xyz=rm_camera_center,
                camera_up_xyz=rm_camera_up,
                camera_right_xyz=rm_camera_right,
                x_resolution=rm_x_resolution,
                y_resolution=rm_y_resolution,
                description=description,
                obstacle_sdf=obstacle_sdf,
                emitter_location=raymarch_emitter_location,
                receiver_locations=raymarch_receiver_locations,
                field_of_view_degrees=rm_fov_deg,
                show_sdf_plane=args.show_sdf_plane,
                show_axes=args.show_axes,
            )

        dataset_folder, dataset_name = os.path.split(args.path_to_dataset)
        assert dataset_name.endswith(".h5")
        dataset_name = dataset_name[: -len(".h5")]
        dataset_size = len(dataset)
        num_digits = math.ceil(math.log10(dataset_size))
        index_str = str(index).zfill(num_digits)

        if not os.path.exists(args.outputpath):
            os.makedirs(args.outputpath)

        filename = (
            f"img_{dataset_name}_{index_str}_{'pred' if prediction else 'gt'}.png"
        )
        filepath = os.path.join(args.outputpath, filename)

        img = torch.clamp(img * 255, min=0.0, max=255.0).to(torch.uint8)

        PIL.Image.fromarray(img.permute(2, 1, 0).cpu().numpy()).save(filepath)


if __name__ == "__main__":
    with torch.no_grad():
        main()
