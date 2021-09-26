import math
import torch
import torch.nn as nn
import torchvision
import numpy as np

from assert_eq import assert_eq
from network_utils import split_network_prediction
from signals_and_geometry import sample_obstacle_map
from simulation_description import SimulationDescription
from which_device import get_compute_device
from utils import progress_bar


def _render_volumetric_slices(
    model,
    recordings,
    obstacle_map,
    description,
    num_splits,
    colour_function,
    locations=None,
):
    with torch.no_grad():
        assert model is None or isinstance(model, nn.Module)
        assert recordings is None or isinstance(recordings, torch.Tensor)
        assert obstacle_map is None or isinstance(obstacle_map, torch.Tensor)
        assert (model is None) != (obstacle_map is None)
        assert (model is None) == (recordings is None)
        assert isinstance(description, SimulationDescription)
        assert isinstance(num_splits, int)

        x_ls = torch.linspace(
            start=description.xmin,
            end=description.xmax,
            steps=description.Nx,
            device=get_compute_device(),
        )
        y_ls = torch.linspace(
            start=description.ymin,
            end=description.ymax,
            steps=description.Ny,
            device=get_compute_device(),
        )

        x_grid, y_grid = torch.meshgrid([x_ls, y_ls])

        num_slices = 10

        slices = []
        for i in range(num_slices):
            t = i / (num_slices - 1)
            z = description.zmin + t * (description.zmax - description.zmin)

            z_grid = z * torch.ones_like(x_grid)
            xyz = torch.stack([x_grid, y_grid, z_grid], dim=2).to(get_compute_device())
            assert_eq(xyz.shape, (description.Nx, description.Ny, 3))
            xyz = xyz.reshape((description.Nx * description.Ny), 3)

            if model is not None:
                prediction = split_network_prediction(
                    model=model,
                    locations=xyz,
                    recordings=recordings,
                    description=description,
                    num_splits=num_splits,
                )
            else:
                prediction = sample_obstacle_map(
                    obstacle_map.unsqueeze(0), xyz.unsqueeze(0), description
                ).squeeze(0)

            assert_eq(prediction.shape, (description.Nx * description.Ny,))
            prediction = prediction.reshape(description.Nx, description.Ny)

            prediction = colour_function(prediction)
            assert_eq(prediction.shape, (3, description.Nx, description.Ny))

            prediction = prediction.cpu()

            if locations is not None:
                for lx, ly, lz in locations:
                    if abs(z - lz) > (
                        (description.zmax - description.zmin) / num_slices
                    ):
                        continue
                    px = round(
                        (
                            (lx - description.xmin)
                            / (description.xmax - description.xmin)
                            * (description.Nx - 1)
                        ).item()
                    )
                    py = round(
                        (
                            (ly - description.ymin)
                            / (description.ymax - description.ymin)
                            * (description.Ny - 1)
                        ).item()
                    )
                    prediction[0, px, py] = 0.0
                    prediction[1, px, py] = 0.0
                    prediction[2, px, py] = 0.0

            slices.append(prediction)

        img_grid = torchvision.utils.make_grid(tensor=slices, nrow=5, pad_value=0.5)
        return img_grid.permute(0, 2, 1)


def render_slices_ground_truth(
    obstacle_map, description, colour_function, locations=None
):
    return _render_volumetric_slices(
        model=None,
        recordings=None,
        obstacle_map=obstacle_map,
        description=description,
        num_splits=1,
        locations=locations,
        colour_function=colour_function,
    )


def render_slices_prediction(
    model, recordings, description, colour_function, num_splits
):
    return _render_volumetric_slices(
        model=model,
        recordings=recordings,
        obstacle_map=None,
        description=description,
        num_splits=num_splits,
        locations=None,
        colour_function=colour_function,
    )


def smoothstep(edge0, edge1, x):
    assert edge0 < edge1
    t = torch.clamp((x - edge0) / (edge1 - edge0), min=0.0, max=1.0)
    return t * t * (3.0 - 2.0 * t)


def blue_orange_sdf_colours(img):
    H, W = img.shape
    img = img.unsqueeze(0)

    def colour(r, g, b):
        return torch.tensor([r, g, b], dtype=torch.float, device=img.device).reshape(
            3, 1, 1
        )

    blue = colour(0.22, 0.33, 0.66)
    orange = colour(0.93, 0.48, 0.10)
    paler_blue = colour(0.50, 0.58, 0.82)
    paler_orange = colour(0.93, 0.87, 0.28)
    white = colour(1.0, 1.0, 1.0)

    sign = torch.sign(img)

    base_colour = blue * (0.5 - 0.5 * sign) + orange * (0.5 + 0.5 * sign)
    paler_colour = paler_blue * (0.5 - 0.5 * sign) + paler_orange * (0.5 + 0.5 * sign)
    mix = torch.exp(-4.0 * torch.abs(img))

    out = base_colour + mix * (paler_colour - base_colour)

    out *= 1.0 - 0.2 * torch.cos(60.0 * img) ** 4

    out = torch.lerp(out, white, 1.0 - smoothstep(0.0, 0.02, torch.abs(img)))

    return out


def colourize_sdf(img):
    return blue_orange_sdf_colours(img)


def is_three_floats(x):
    return len(x) == 3 and all([isinstance(xi, float) for xi in x])


def vector_cross(a, b):
    assert is_three_floats(a)
    assert is_three_floats(b)
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def vector_length(x):
    assert is_three_floats(x)
    return math.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)


def vector_normalize(v, norm=1.0):
    assert is_three_floats(v)
    k = norm / vector_length(v)
    return [k * v[0], k * v[1], k * v[2]]


def _simulation_boundary_sdf(description, sample_locations, radius):
    assert isinstance(description, SimulationDescription)
    assert isinstance(sample_locations, torch.Tensor)
    D, N, M = sample_locations.shape
    assert D == 3
    assert isinstance(radius, float)

    locations_x = sample_locations[0]
    locations_y = sample_locations[1]
    locations_z = sample_locations[2]

    # --- x axes ---

    # compress x axis
    x_axes_locations_x_positive = torch.clamp(locations_x - description.xmax, min=0.0)
    x_axes_locations_x_negative = torch.clamp(locations_x - description.xmin, max=0.0)
    x_axes_locations_x = x_axes_locations_x_positive + x_axes_locations_x_negative

    # mirror and shift y axis
    x_axes_locations_y = torch.minimum(
        torch.abs(locations_y - description.ymin),
        torch.abs(locations_y - description.ymax),
    )

    # mirror and shift z axis
    x_axes_locations_z = torch.minimum(
        torch.abs(locations_z - description.zmin),
        torch.abs(locations_z - description.zmax),
    )

    # distance to point
    x_axes_locations = torch.stack(
        [x_axes_locations_x, x_axes_locations_y, x_axes_locations_z], dim=0
    )
    sdf_x_axes = torch.norm(x_axes_locations, dim=0) - radius

    # --- y axes ---

    # mirror and shift x axis
    y_axes_locations_x = torch.minimum(
        torch.abs(locations_x - description.xmin),
        torch.abs(locations_x - description.xmax),
    )

    # compress y axis
    y_axes_locations_y_positive = torch.clamp(locations_y - description.ymax, min=0.0)
    y_axes_locations_y_negative = torch.clamp(locations_y - description.ymin, max=0.0)
    y_axes_locations_y = y_axes_locations_y_positive + y_axes_locations_y_negative

    # mirror and shift z axis
    y_axes_locations_z = torch.minimum(
        torch.abs(locations_z - description.zmin),
        torch.abs(locations_z - description.zmax),
    )

    # distance to point
    y_axes_locations = torch.stack(
        [y_axes_locations_x, y_axes_locations_y, y_axes_locations_z], dim=0
    )
    sdf_y_axes = torch.norm(y_axes_locations, dim=0) - radius

    # --- x axes ---

    # mirror and shift z axis
    z_axes_locations_x = torch.minimum(
        torch.abs(locations_x - description.xmin),
        torch.abs(locations_x - description.xmax),
    )

    # mirror and shift y axis
    z_axes_locations_y = torch.minimum(
        torch.abs(locations_y - description.ymin),
        torch.abs(locations_y - description.ymax),
    )

    # compress x axis
    z_axes_locations_z_positive = torch.clamp(locations_z - description.zmax, min=0.0)
    z_axes_locations_z_negative = torch.clamp(locations_z - description.zmin, max=0.0)
    z_axes_locations_z = z_axes_locations_z_positive + z_axes_locations_z_negative

    # distance to point
    z_axes_locations = torch.stack(
        [z_axes_locations_x, z_axes_locations_y, z_axes_locations_z], dim=0
    )
    sdf_z_axes = torch.norm(z_axes_locations, dim=0) - radius

    return torch.minimum(
        torch.minimum(
            sdf_x_axes,
            sdf_y_axes,
        ),
        sdf_z_axes,
    )


def _spheres_sdf(sphere_locations, sample_locations, sphere_radius):
    assert isinstance(sphere_locations, torch.Tensor)
    D, L = sphere_locations.shape
    assert D == 3
    assert isinstance(sample_locations, torch.Tensor)
    D, N, M = sample_locations.shape
    assert D == 3
    assert isinstance(sphere_radius, float)
    sqr_dists = torch.sum(
        torch.square(
            sphere_locations.reshape(3, 1, 1, L) - sample_locations.reshape(3, N, M, 1)
        ),
        dim=0,
    )
    assert_eq(sqr_dists.shape, (N, M, L))
    min_dist = torch.sqrt(
        torch.min(
            sqr_dists,
            dim=-1,
        )[0]
    )
    assert_eq(min_dist.shape, (N, M))
    return min_dist - sphere_radius


def _raymarch_sdf_impl(
    camera_center_xyz,
    camera_up_xyz,
    camera_right_xyz,
    x_resolution,
    y_resolution,
    description,
    obstacle_sdf,
    model,
    recordings,
    receiver_locations,
    emitter_location,
    field_of_view_degrees,
    num_splits,
):
    with torch.no_grad():
        assert is_three_floats(camera_center_xyz)
        assert is_three_floats(camera_up_xyz)
        assert is_three_floats(camera_right_xyz)
        assert isinstance(description, SimulationDescription)
        assert isinstance(num_splits, int)
        assert isinstance(field_of_view_degrees, float)
        if obstacle_sdf is not None:
            assert isinstance(obstacle_sdf, torch.Tensor)
            assert obstacle_sdf.shape == (
                description.Nx,
                description.Ny,
                description.Nz,
            )
            assert model is None
            assert recordings is None
            prediction = False
        else:
            assert isinstance(model, nn.Module)
            assert isinstance(recordings, torch.Tensor)
            assert len(recordings.shape) == 2
            assert recordings.shape[1] == description.output_length
            prediction = True

        show_emitter = emitter_location is not None
        show_receivers = receiver_locations is not None

        if show_emitter:
            assert isinstance(emitter_location, torch.Tensor)
        if show_receivers:
            isinstance(receiver_locations, torch.Tensor)

        # create grid of sampling points using meshgrid between two camera directions
        def make_tensor_3f(t, normalize=False):
            ret = torch.tensor(
                [*t], dtype=torch.float32, device=get_compute_device()
            ).reshape(3, 1, 1)
            if normalize:
                return ret / torch.norm(ret, dim=0, keepdim=True)
            return ret

        camera_center = make_tensor_3f(camera_center_xyz)
        camera_up = make_tensor_3f(camera_up_xyz)
        camera_right = make_tensor_3f(camera_right_xyz)
        camera_forward = make_tensor_3f(
            vector_cross(camera_up_xyz, camera_right_xyz), normalize=True
        )

        # create grid of view vectors using cross of two camera directions (and maybe offset from center for slight perspective)
        ls_x = torch.linspace(
            start=-1.0, end=1.0, steps=x_resolution, device=get_compute_device()
        )
        ls_y = torch.linspace(
            start=-1.0, end=1.0, steps=y_resolution, device=get_compute_device()
        )
        grid_x, grid_y = torch.meshgrid(ls_x, ls_y)
        offsets_x = grid_x.unsqueeze(0) * camera_right
        offsets_y = grid_y.unsqueeze(0) * camera_up
        locations = camera_center + offsets_x + offsets_y

        directions = camera_forward.repeat(1, x_resolution, y_resolution)

        # Add perspective distortion
        directions = directions + math.tan(field_of_view_degrees * math.pi / 180.0) * (
            offsets_x + offsets_y
        )
        directions /= torch.norm(directions, dim=0, keepdim=True)

        def _sample_obstacle_sdf(l):
            assert_eq(l.shape, (3, x_resolution, y_resolution))
            l_flat = l.reshape(1, 3, x_resolution * y_resolution).permute(0, 2, 1)
            assert l_flat.shape == (1, x_resolution * y_resolution, 3)
            if prediction:
                # num_splits = 256  # 128
                split_size = (x_resolution * y_resolution) // num_splits
                values_acc = []
                for i in range(num_splits):
                    idx_lo = i * split_size
                    idx_hi = (i + 1) * split_size
                    values_acc.append(
                        model(recordings.unsqueeze(0), l_flat[:, idx_lo:idx_hi])
                    )
                sdf_values = torch.cat(values_acc, dim=1)
            else:
                sdf_values = sample_obstacle_map(
                    obstacle_map_batch=obstacle_sdf.unsqueeze(0),
                    locations_xyz_batch=l_flat,
                    description=description,
                )
            assert_eq(sdf_values.shape, (1, x_resolution * y_resolution))
            sdf_values = sdf_values.reshape(x_resolution, y_resolution)
            return sdf_values

        # keep a boolean mask of rays that have not yet collided
        active = torch.ones(
            (x_resolution, y_resolution), dtype=torch.bool, device=get_compute_device()
        )

        hit_axes = torch.zeros(
            (x_resolution, y_resolution), dtype=torch.bool, device=get_compute_device()
        )

        if show_emitter:
            hit_emitter = torch.zeros(
                (x_resolution, y_resolution),
                dtype=torch.bool,
                device=get_compute_device(),
            )

        if show_receivers:
            hit_receivers = torch.zeros(
                (x_resolution, y_resolution),
                dtype=torch.bool,
                device=get_compute_device(),
            )

        num_iterations = 64
        for i in range(num_iterations):
            # cheap approximation for outer SDF:
            # - clamp locations to inner volume
            # - sample SDF values at clamped locations
            # - add back distance added due to clamping
            # - apply a fudge factor to safely account for slight errors

            original_locations = locations.clone()

            locations[0].clamp_(min=description.xmin, max=description.xmax)
            locations[1].clamp_(min=description.ymin, max=description.ymax)
            locations[2].clamp_(min=description.zmin, max=description.zmax)

            clamp_displacement = torch.sqrt(
                torch.sum(torch.square(locations - original_locations), dim=0)
            )
            clamp_displacement *= 0.8

            # get SDF values at each ray location
            sampled_sdf_obstacles = _sample_obstacle_sdf(locations) + clamp_displacement

            sampled_sdf_obstacles.nan_to_num_(nan=np.inf)

            sampled_sdf_axes = _simulation_boundary_sdf(
                description, original_locations, radius=0.001
            )

            sampled_sdf = torch.minimum(sampled_sdf_obstacles, sampled_sdf_axes)

            if show_emitter:
                sampled_sdf_emitter = _spheres_sdf(
                    emitter_location.unsqueeze(-1), original_locations, 0.01
                )

                # sampled_sdf.clamp_(max=sampled_sdf_emitter)
                sampled_sdf = torch.minimum(sampled_sdf, sampled_sdf_emitter)

            if show_receivers:
                sampled_sdf_receivers = _spheres_sdf(
                    receiver_locations, original_locations, 0.01
                )

                # sampled_sdf.clamp_(max=sampled_sdf_receivers)
                sampled_sdf = torch.minimum(sampled_sdf, sampled_sdf_receivers)

            locations = original_locations

            # if SDF value is below threshold, make inactive
            threshold = 0.001
            active[sampled_sdf <= threshold] = 0
            hit_axes[sampled_sdf_axes <= threshold] = 1

            if show_emitter:
                hit_emitter[sampled_sdf_emitter <= threshold] = 1
            if show_receivers:
                hit_receivers[sampled_sdf_receivers <= threshold] = 1

            # advance all active rays by their direction vector times their SDF value
            locations[:, active] += (sampled_sdf * directions)[:, active]

            progress_bar(i, num_iterations)

        ret = torch.zeros(
            (3, x_resolution, y_resolution),
            dtype=torch.float32,
            device=get_compute_device(),
        )

        # fill non-collided pixels with background colour
        ret[:, active] = 1.0

        inactive = active.logical_not()

        # colour hit stuff with obstacle colour
        ret[0][inactive] = 0.8
        ret[1][inactive] = 0.8
        ret[2][inactive] = 0.8

        # colour axes
        ret[0][hit_axes] = 0.0
        ret[1][hit_axes] = 0.0
        ret[2][hit_axes] = 0.0

        # colour emitter
        if show_emitter:
            ret[0][hit_emitter] = 0.0
            ret[1][hit_emitter] = 0.0
            ret[2][hit_emitter] = 1.0

        # colour receivers
        if show_receivers:
            ret[0][hit_receivers] = 1.0
            ret[1][hit_receivers] = 0.5
            ret[2][hit_receivers] = 0.0

        # shade collide pixels with x,y,z partial derivatives of SDF at sampling locations
        def combined_sdf(loc):
            v = _sample_obstacle_sdf(loc)
            if show_emitter:
                v2 = _spheres_sdf(emitter_location.unsqueeze(-1), loc, 0.01)
                v = torch.minimum(v, v2)
            if show_receivers:
                v2 = _spheres_sdf(receiver_locations, loc, 0.01)
                v = torch.minimum(v, v2)
            return v

        h = 0.02
        dx = make_tensor_3f([0.5 * h, 0.0, 0.0])
        dy = make_tensor_3f([0.0, 0.5 * h, 0.0])
        dz = make_tensor_3f([0.0, 0.0, 0.5 * h])

        dsdfdx = (1.0 / h) * (
            combined_sdf(locations + dx) - combined_sdf(locations - dx)
        )
        dsdfdy = (1.0 / h) * (
            combined_sdf(locations + dy) - combined_sdf(locations - dy)
        )
        dsdfdz = (1.0 / h) * (
            combined_sdf(locations + dz) - combined_sdf(locations - dz)
        )
        assert_eq(dsdfdx.shape, (x_resolution, y_resolution))
        assert_eq(dsdfdy.shape, (x_resolution, y_resolution))
        assert_eq(dsdfdz.shape, (x_resolution, y_resolution))
        sdf_normal = torch.stack([dsdfdx, dsdfdy, dsdfdz], dim=0)
        sdf_normal /= torch.clamp(torch.norm(sdf_normal, dim=0, keepdim=True), min=1e-3)
        assert_eq(sdf_normal.shape, (3, x_resolution, y_resolution))
        light_dir = make_tensor_3f([-0.25, -1.0, 0.5], normalize=True)
        normal_dot_light = torch.sum(sdf_normal * light_dir, dim=0)
        assert_eq(normal_dot_light.shape, (x_resolution, y_resolution))
        shading = 0.2 + 0.8 * torch.clamp(normal_dot_light, min=0.0)

        ret[0][inactive] *= shading[inactive]
        ret[1][inactive] *= shading[inactive]
        ret[2][inactive] *= shading[inactive]

        return ret


def raymarch_sdf_ground_truth(
    camera_center_xyz,
    camera_up_xyz,
    camera_right_xyz,
    x_resolution,
    y_resolution,
    description,
    obstacle_sdf,
    receiver_locations,
    emitter_location,
    field_of_view_degrees,
    num_splits,
):
    return _raymarch_sdf_impl(
        camera_center_xyz=camera_center_xyz,
        camera_up_xyz=camera_up_xyz,
        camera_right_xyz=camera_right_xyz,
        x_resolution=x_resolution,
        y_resolution=y_resolution,
        description=description,
        obstacle_sdf=obstacle_sdf,
        model=None,
        recordings=None,
        receiver_locations=receiver_locations,
        emitter_location=emitter_location,
        field_of_view_degrees=field_of_view_degrees,
        num_splits=num_splits,
    )


def raymarch_sdf_prediction(
    camera_center_xyz,
    camera_up_xyz,
    camera_right_xyz,
    x_resolution,
    y_resolution,
    description,
    model,
    recordings,
    receiver_locations,
    emitter_location,
    field_of_view_degrees,
    num_splits,
):
    return _raymarch_sdf_impl(
        camera_center_xyz=camera_center_xyz,
        camera_up_xyz=camera_up_xyz,
        camera_right_xyz=camera_right_xyz,
        x_resolution=x_resolution,
        y_resolution=y_resolution,
        description=description,
        obstacle_sdf=None,
        model=model,
        recordings=recordings,
        receiver_locations=receiver_locations,
        emitter_location=emitter_location,
        field_of_view_degrees=field_of_view_degrees,
        num_splits=num_splits,
    )
