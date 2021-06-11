from config import (
    OutputConfig,
)
from config_constants import (
    output_format_depthmap,
    output_format_heatmap,
    output_format_sdf,
)
from featurize import make_dense_outputs
import torch


def convert_sdf_to_occupancy(sdf):
    """
    SDF values less than `threshold` are considered to be inside an object
    """
    threshold = 0.0
    B, C, H, W = sdf.shape
    assert B == 1
    assert C in [1, 2]
    assert H == W
    return (sdf[:, :1, :, :] <= threshold).to(torch.bool)


def convert_heatmap_to_occupancy(hm):
    """
    heatmap values greater than `threshold` are considered to be inside an object
    """
    threshold = 0.5
    B, C, H, W = hm.shape
    assert B == 1
    assert C in [1, 2]
    assert H == W
    return (hm[:, :1, :, :] >= threshold).to(torch.bool)


def convert_occupancy_to_shadowed_occupancy(occupancy):
    B, C, H, W = occupancy.shape
    assert B == 1
    assert C == 1
    assert H == W
    mask_row = torch.zeros((1, 1, W), dtype=torch.bool, device=occupancy.device)
    out = torch.zeros((1, 1, H, W), dtype=torch.bool, device=occupancy.device)
    for i in list(range(H))[::-1]:
        mask_row[occupancy[:, :, i, :] > 0.0] = 1
        out[:, :, i, :] = mask_row
    return out


def convert_depthmap_to_shadowed_occupancy(depthmap):
    B, C, W = depthmap.shape
    assert B == 1
    assert C in [1, 2]
    out = torch.zeros((1, 1, W, W), dtype=torch.bool, device=depthmap.device)
    for b in range(B):
        for x in range(W):
            d = depthmap[b, 0, x].item()
            dc = min(max(d, 0.0), 1.0)
            i = int((1.0 - dc) * (W - 1))
            out[b, 0, 0:i, x] = 1
    return out


def predicted_occupancy(tensor, output_config, shadowed):
    assert isinstance(output_config, OutputConfig)
    if output_config.format == output_format_depthmap:
        assert shadowed
        return convert_depthmap_to_shadowed_occupancy(tensor)
    elif output_config.format == output_format_heatmap:
        o = convert_heatmap_to_occupancy(tensor)
    elif output_config.format == output_format_sdf:
        o = convert_sdf_to_occupancy(tensor)
    else:
        raise Exception("Unrecognized output format")
    if shadowed:
        return convert_occupancy_to_shadowed_occupancy(o)
    return o


def ground_truth_occupancy(obstacles, size, shadowed):
    sdf = make_dense_outputs(obstacles, output_format_sdf, size)
    o = convert_sdf_to_occupancy(sdf.unsqueeze(0).unsqueeze(0))
    if shadowed:
        return convert_occupancy_to_shadowed_occupancy(o)
    return o


def compute_error_metrics(occupancy_gt, occupancy_pred):
    assert isinstance(occupancy_gt, torch.BoolTensor) or isinstance(
        occupancy_gt, torch.cuda.BoolTensor
    )
    assert isinstance(occupancy_pred, torch.BoolTensor) or isinstance(
        occupancy_pred, torch.cuda.BoolTensor
    )
    assert occupancy_gt.shape == occupancy_pred.shape
    B, C, H, W = occupancy_gt.shape
    assert B == 1
    assert C == 1
    assert H == W

    gt_true = occupancy_gt
    gt_false = torch.logical_not(occupancy_gt)
    pred_true = occupancy_pred
    pred_false = torch.logical_not(occupancy_pred)

    def as_fraction(t):
        assert isinstance(t, torch.BoolTensor) or isinstance(t, torch.cuda.BoolTensor)
        assert t.shape == (1, 1, H, H)
        f = torch.sum(t).item() / (H ** 2)
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
