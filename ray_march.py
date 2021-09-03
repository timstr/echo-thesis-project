import fix_dead_command_line

import os
import torch
import math
from argparse import ArgumentParser
import PIL

from network_utils import split_network_prediction
from visualization import (
    raymarch_sdf_ground_truth,
    raymarch_sdf_prediction,
    vector_cross,
    vector_length,
    vector_normalize,
)
from signals_and_geometry import convolve_recordings, make_fm_chirp, obstacle_map_to_sdf
from split_till_it_fits import SplitSize, split_till_it_fits
from assert_eq import assert_eq
from the_device import the_device
from current_simulation_description import (
    all_grid_locations,
    make_receiver_indices,
    make_simulation_description,
    minimum_x_units,
)
from dataset3d import WaveDataset3d, k_sensor_recordings, k_sdf
from time_of_flight_net import TimeOfFlightNet
from torch_utils import restore_module


def main():
    parser = ArgumentParser()
    parser.add_argument("path_to_dataset", type=str)
    parser.add_argument("indices", nargs="*", metavar="indices", type=int, default=0)
    parser.add_argument(
        "--tofcropsize",
        type=int,
        dest="tofcropsize",
        default=None,
        help="Number of samples used in time-of-flight crop",
    )
    parser.add_argument(
        "--chirpf0",
        type=float,
        dest="chirpf0",
        default=None,
        help="chirp start frequency (Hz)",
    )
    parser.add_argument(
        "--chirpf1",
        type=float,
        dest="chirpf1",
        default=None,
        help="chirp end frequency (Hz)",
    )
    parser.add_argument(
        "--chirplen",
        type=float,
        dest="chirplen",
        default=None,
        help="chirp duration (seconds)",
    )
    parser.add_argument(
        "--receivercountx", type=int, dest="receivercountx", default=None
    )
    parser.add_argument(
        "--receivercounty", type=int, dest="receivercounty", default=None
    )
    parser.add_argument(
        "--receivercountz", type=int, dest="receivercountz", default=None
    )
    parser.add_argument(
        "--restoremodelpath", type=str, dest="restoremodelpath", default=None
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
    parser.add_argument("--supersampling", type=int, dest="supersampling", default=None)
    args = parser.parse_args()

    if args.restoremodelpath is None:
        prediction = False
        assert args.tofcropsize is None
        assert args.chirpf0 is None
        assert args.chirpf1 is None
        assert args.chirplen is None
        if not args.showsensor:
            assert args.receivercountx is None
            assert args.receivercounty is None
            assert args.receivercountz is None
        assert args.precompute == False
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

    sensor_indices = make_receiver_indices(
        args.receivercountx,
        args.receivercounty,
        args.receivercountz,
    )

    if args.showsensor:
        raymarch_emitter_location = torch.tensor(description.emitter_location).to(
            the_device
        )
        assert_eq(raymarch_emitter_location.shape, (3,))
        raymarch_receiver_locations = (
            torch.tensor(description.sensor_locations[sensor_indices])
            .permute(1, 0)
            .to(the_device)
        )
        assert_eq(raymarch_receiver_locations.shape, (3, len(sensor_indices)))
    else:
        raymarch_emitter_location = None
        raymarch_receiver_locations = None

    if prediction:

        the_model = TimeOfFlightNet(
            speed_of_sound=description.air_properties.speed_of_sound,
            sampling_frequency=description.output_sampling_frequency,
            recording_length_samples=description.output_length,
            crop_length_samples=args.tofcropsize,
            emitter_location=description.emitter_location,
            receiver_locations=description.sensor_locations[sensor_indices],
        ).to("cuda")
        restore_module(the_model, args.restoremodelpath)
        the_model.eval()

        fm_chirp = (
            torch.tensor(
                make_fm_chirp(
                    begin_frequency_Hz=args.chirpf0,
                    end_frequency_Hz=args.chirpf1,
                    sampling_frequency=description.output_sampling_frequency,
                    chirp_length_samples=math.ceil(
                        args.chirplen * description.output_sampling_frequency
                    ),
                    wave="sine",
                )
            )
            .float()
            .to(the_device)
        )

    for index in args.indices:
        print(f"Rendering example {index}")
        example = dataset[index]

        # rm_camera_center = [description.xmin - 0.01, 0.0, 0.0]
        # rm_camera_up = [0.0, 0.5 * description.Ny * description.dy, 0.0]
        # rm_camera_right = [0.0, 0.0, 0.5 * description.Nz * description.dz]
        # rm_x_resolution = 512
        # rm_y_resolution = 512
        # rm_fov_deg = 30.0

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

        if prediction:
            recordings_ir = example[k_sensor_recordings][sensor_indices].to(the_device)

            recordings_fm = convolve_recordings(fm_chirp, recordings_ir, description)

            if args.precompute:
                locations = all_grid_locations(
                    the_device, description, downsample_factor=1
                )

                x_steps = description.Nx - minimum_x_units
                y_steps = description.Ny
                z_steps = description.Nz

                offset = -0.5 * (supersampling - 1) / supersampling

                sdf_pred_acc = torch.zeros(
                    (x_steps * y_steps * z_steps,), device=the_device
                )

                for ss in range(supersampling ** 3):
                    ss_i = ss // supersampling ** 2
                    ss_j = ss // supersampling % supersampling
                    ss_k = ss % supersampling

                    ss_dx = description.dx * (offset + (ss_i / supersampling))
                    ss_dy = description.dy * (offset + (ss_j / supersampling))
                    ss_dz = description.dz * (offset + (ss_k / supersampling))

                    locations_offset = locations + torch.tensor(
                        [[ss_dx, ss_dy, ss_dz]], device=the_device
                    )

                    sdf_pred_acc += split_till_it_fits(
                        split_network_prediction,
                        network_prediction_split_size,
                        model=the_model,
                        locations=locations_offset,
                        recordings=recordings_fm,
                        description=description,
                        show_progress_bar=True,
                    )

                sdf_pred = sdf_pred_acc / (supersampling ** 3)
                sdf_pred = sdf_pred.reshape(x_steps, y_steps, z_steps)

                obstacle_map = torch.zeros(
                    (description.Nx, description.Ny, description.Nz),
                    device=the_device,
                    dtype=torch.bool,
                )
                obstacle_map[minimum_x_units:] = sdf_pred <= 0.0

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
                    recordings=recordings_fm,
                    emitter_location=raymarch_emitter_location,
                    receiver_locations=raymarch_receiver_locations,
                    field_of_view_degrees=rm_fov_deg,
                )
        else:
            obstacle_sdf = example[k_sdf].cuda()

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
            )

        dataset_folder, dataset_name = os.path.split(args.path_to_dataset)
        assert dataset_name.endswith(".h5")
        dataset_name = dataset_name[: -len(".h5")]
        dataset_size = len(dataset)
        num_digits = math.ceil(math.log10(dataset_size))
        index_str = str(index).zfill(num_digits)

        filename = (
            f"img_{dataset_name}_{index_str}_{'pred' if prediction else 'gt'}.png"
        )

        img = torch.clamp(img * 255, min=0.0, max=255.0).to(torch.uint8)

        PIL.Image.fromarray(img.permute(2, 1, 0).cpu().numpy()).save(filename)


if __name__ == "__main__":
    with torch.no_grad():
        main()
