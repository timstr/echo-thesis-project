import torch
import torch.nn as nn

from assert_eq import assert_eq
from utils import progress_bar
from simulation_description import SimulationDescription
from current_simulation_description import minimum_x_units


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


def evaluate_prediction(sdf_pred, sdf_gt, description, downsample_factor):
    assert isinstance(sdf_pred, torch.Tensor)
    assert isinstance(sdf_gt, torch.Tensor)
    assert isinstance(description, SimulationDescription)
    assert isinstance(downsample_factor, int)
    assert downsample_factor >= 1
    x_steps = (description.Nx - minimum_x_units) // downsample_factor
    y_steps = description.Ny // downsample_factor
    z_steps = description.Nz // downsample_factor
    assert_eq(sdf_pred.shape, (x_steps, y_steps, z_steps))
    assert_eq(sdf_gt.shape, (x_steps, y_steps, z_steps))

    mse_sdf = torch.mean((sdf_pred - sdf_gt) ** 2).item()

    occupancy_pred = (sdf_pred <= 0.0).bool()
    occupancy_gt = (sdf_gt <= 0.0).bool()

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
        "mean_squared_error_sdf": mse_sdf,
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
