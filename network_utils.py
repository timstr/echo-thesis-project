from dataset_adapters import (
    wavesim_to_batgnet_occupancy,
    wavesim_to_batgnet_spectrogram,
    wavesim_to_batvision_depthmap,
    wavesim_to_batvision_spectrogram,
    wavesim_to_batvision_waveform,
)
import numpy as np
import torch
import torch.nn as nn

from signals_and_geometry import (
    backfill_occupancy,
    sample_obstacle_map,
    sdf_to_occupancy,
)
from split_till_it_fits import SplitSize, split_till_it_fits
from assert_eq import assert_eq
from utils import progress_bar
from simulation_description import SimulationDescription
from current_simulation_description import all_grid_locations, minimum_x_units
from which_device import get_compute_device
from dataset3d import k_sdf, k_sensor_recordings


def split_network_prediction(
    model, locations, recordings, description, num_splits, show_progress_bar=False
):
    with torch.no_grad():
        assert isinstance(model, nn.Module)
        assert isinstance(locations, torch.Tensor)
        assert isinstance(recordings, torch.Tensor)
        assert isinstance(description, SimulationDescription)
        assert isinstance(num_splits, int)
        N, D = locations.shape
        locations = locations.reshape(1, N, D)
        assert_eq(D, 3)
        R, L = recordings.shape
        assert_eq(L, description.output_length)
        recordings = recordings.reshape(1, R, L)
        splits = []
        for i in range(num_splits):
            split_lo = N * i // num_splits
            split_hi = N * (i + 1) // num_splits
            xyz_split = locations[:, split_lo:split_hi]
            prediction_split = model(recordings=recordings, sample_locations=xyz_split)
            splits.append(prediction_split)
            if show_progress_bar:
                progress_bar(i, num_splits)
        prediction = torch.cat(splits, dim=1).squeeze(0)
        assert_eq(prediction.shape, (N,))
        return prediction


def evaluate_prediction(occupancy_pred, occupancy_gt):
    assert isinstance(occupancy_pred, torch.Tensor)
    assert_eq(occupancy_pred.dtype, torch.bool)
    assert isinstance(occupancy_gt, torch.Tensor)
    assert_eq(occupancy_gt.dtype, torch.bool)

    gt_true = occupancy_gt
    gt_false = torch.logical_not(occupancy_gt)
    pred_true = occupancy_pred
    pred_false = torch.logical_not(occupancy_pred)

    def as_fraction(t):
        assert isinstance(t, torch.BoolTensor) or isinstance(t, torch.cuda.BoolTensor)
        f = torch.mean(t.float()).item()
        assert f >= 0.0 and f <= 1.0
        return f

    intersection = torch.logical_and(gt_true, pred_true)
    union = torch.logical_or(gt_true, pred_true)

    f_intersection = as_fraction(intersection)
    f_union = as_fraction(union)

    assert f_intersection >= 0.0
    assert f_intersection <= 1.0
    assert f_union >= 0.0
    assert f_union <= 1.0
    assert f_intersection <= f_union

    epsilon = 1e-6

    intersection_over_union = (f_intersection / f_union) if (f_union > epsilon) else 1.0

    assert intersection_over_union <= 1.0

    true_positives = torch.logical_and(gt_true, pred_true)
    true_negatives = torch.logical_and(gt_false, pred_false)
    false_positives = torch.logical_and(gt_false, pred_true)
    false_negatives = torch.logical_and(gt_true, pred_false)

    f_true_positives = as_fraction(true_positives)
    f_true_negatives = as_fraction(true_negatives)
    f_false_positives = as_fraction(false_positives)
    f_false_negatives = as_fraction(false_negatives)

    assert (
        abs(
            f_true_positives
            + f_true_negatives
            + f_false_positives
            + f_false_negatives
            - 1.0
        )
        < epsilon
    )

    selected = f_true_positives + f_false_positives
    relevant = f_true_positives + f_false_negatives

    precision = f_true_positives / selected if (selected > epsilon) else 0.0
    recall = f_true_positives / relevant if (relevant > epsilon) else 0.0

    f1score = (
        (2.0 * precision * recall / (precision + recall))
        if (abs(precision + recall) > epsilon)
        else 0.0
    )

    return {
        "intersection": f_intersection,
        "union": f_union,
        "intersection_over_union": intersection_over_union,
        "true_positives": f_true_positives,
        "true_negatives": f_true_negatives,
        "false_positives": f_false_positives,
        "false_negatives": f_false_negatives,
        "precision": precision,
        "recall": recall,
        "f1score": f1score,
    }


def evaluate_tofnet_on_whole_dataset(
    the_model,
    dataset,
    description,
    validationdownsampling,
    adapt_signals_fn,
    sdf_offset,
    backfill,
    no_sdf,
    split_size,
):
    assert isinstance(the_model, nn.Module)
    assert isinstance(description, SimulationDescription)
    assert isinstance(validationdownsampling, int)
    assert isinstance(sdf_offset, float)
    assert isinstance(backfill, bool)
    assert isinstance(no_sdf, bool)
    assert isinstance(split_size, SplitSize)
    with torch.no_grad():
        total_metrics = {}
        locations = all_grid_locations(
            get_compute_device(), description, downsample_factor=validationdownsampling
        )
        x_steps = (description.Nx - minimum_x_units) // validationdownsampling
        y_steps = description.Ny // validationdownsampling
        z_steps = description.Nz // validationdownsampling
        N = len(dataset)
        for i, dd in enumerate(dataset):
            example = adapt_signals_fn(dd.to(get_compute_device()))

            sdf_gt = sample_obstacle_map(
                obstacle_map_batch=example[k_sdf].to(get_compute_device()),
                locations_xyz_batch=locations,
                description=description,
            )
            if no_sdf:
                sdf_gt = -1.0 + 2.0 * (sdf_gt > 0.0).float()

            sdf_pred = split_till_it_fits(
                split_network_prediction,
                split_size,
                model=the_model,
                locations=locations,
                recordings=example[k_sensor_recordings],
                description=description,
            )

            sdf_pred = sdf_pred.reshape(x_steps, y_steps, z_steps)
            sdf_gt = sdf_gt.reshape(x_steps, y_steps, z_steps)

            occupancy_gt = sdf_to_occupancy(sdf_gt)
            occupancy_pred = sdf_to_occupancy(sdf_pred, threshold=sdf_offset)

            if backfill:
                occupancy_gt = backfill_occupancy(occupancy_gt)
                occupancy_pred = backfill_occupancy(occupancy_pred)

            metrics = evaluate_prediction(
                occupancy_gt=occupancy_gt, occupancy_pred=occupancy_pred
            )

            mae_sdf = torch.mean(torch.abs(sdf_gt - sdf_pred)).item()

            metrics["mean_absolute_error_sdf"] = mae_sdf

            assert isinstance(metrics, dict)
            for k, v in metrics.items():
                assert isinstance(v, float)
                if not k in total_metrics:
                    total_metrics[k] = []
                total_metrics[k].append(v)
            progress_bar(i, N)

        mean_metrics = {}
        for k, v in total_metrics.items():
            mean_metrics[k] = np.mean(v)

        return mean_metrics


def evaluate_batvision_on_whole_dataset(
    the_model,
    dataset,
    description,
    adapt_signals_fn,
    batvision_mode,
):
    assert isinstance(the_model, nn.Module)
    assert isinstance(description, SimulationDescription)
    assert batvision_mode in ["waveform", "spectrogram"]
    with torch.no_grad():
        all_errors = []
        N = len(dataset)
        for i, dd in enumerate(dataset):
            example = adapt_signals_fn(dd.to(get_compute_device()))
            if batvision_mode == "waveform":
                inputs = wavesim_to_batvision_waveform(example)
            else:
                inputs = wavesim_to_batvision_spectrogram(example)

            depthmap_gt = wavesim_to_batvision_depthmap(example)
            assert_eq(depthmap_gt.shape, (128, 128))

            depthmap_pred = the_model(inputs.unsqueeze(0)).squeeze(0)
            assert_eq(depthmap_pred.shape, (128, 128))

            error = torch.mean(torch.abs(depthmap_pred - depthmap_gt))

            all_errors.append(error.item())

            progress_bar(i, N)

        return np.mean(all_errors)


def evaluate_batgnet_on_whole_dataset(
    the_model, dataset, description, adapt_signals_fn, backfill
):
    assert isinstance(the_model, nn.Module)
    assert isinstance(description, SimulationDescription)
    assert isinstance(backfill, bool)
    with torch.no_grad():
        all_errors = []
        N = len(dataset)
        for i, dd in enumerate(dataset):
            example = adapt_signals_fn(dd.to(get_compute_device()))
            spectrograms = wavesim_to_batgnet_spectrogram(example)

            occupancy_gt = (
                wavesim_to_batgnet_occupancy(example, backfill).squeeze(0).float()
            )
            assert_eq(occupancy_gt.shape, (64, 64, 64))

            occupancy_pred = the_model(spectrograms.unsqueeze(0)).squeeze(0)
            assert_eq(occupancy_pred.shape, (64, 64, 64))

            error = torch.mean(torch.square(occupancy_pred - occupancy_gt))

            all_errors.append(error.item())

            progress_bar(i, N)

        return np.mean(all_errors)
