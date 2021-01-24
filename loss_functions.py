from device_dict import DeviceDict
import torch
import math
import numpy as np

from EchoLearnNN import EchoLearnNN
from featurize import make_deterministic_validation_batches_implicit
from the_device import the_device, what_my_gpu_can_handle
from progress_bar import progress_bar

def meanSquaredErrorLoss(batch_gt, batch_pred):
    assert isinstance(batch_gt, DeviceDict)
    assert isinstance(batch_pred, DeviceDict)
    x1 = batch_gt["output"]
    x2 = batch_pred["output"][:, 0]
    assert x1.shape == x2.shape
    mse = torch.nn.functional.mse_loss(x1, x2)
    terms = { "mean_squared_error": mse }
    return mse, terms

def meanAndVarianceLoss(batch_gt, batch_pred):
    assert isinstance(batch_gt, DeviceDict)
    assert isinstance(batch_pred, DeviceDict)
    y = batch_gt["output"]
    z_hat = batch_pred["output"]
    y_hat = z_hat[:, 0]
    sigma_hat = z_hat[:, 1]

    sqrt2pi = math.sqrt(2.0 * np.pi)
    squared_error = (y - y_hat)**2

    #     phi(y|x)  = exp(-(y - y_hat)^2/(2*sigma^2))
    #                     / (sqrt(2*pi)*sigma)
    # log(phi(y|x)) = -(y - y_hat)^2 / (2*sigma^2)
    #                     - log(sqrt(2*pi)*sigma)
    #               = -0.5 * (y - y_hat)^2 / sigma^2
    #                     - (log(sqrt(2*pi)) + log(sigma))
    #               = -0.5 * (y - y_hat)^2 * (1/sigma)^2
    #                     - (log(sqrt(2*pi)) - log(1/sigma))

    sigma_hat_plus_epsilon = sigma_hat + 1e-2
    log_numerator = -0.5 * squared_error / sigma_hat_plus_epsilon**2
    log_denominator = math.log(sqrt2pi) + torch.log(sigma_hat_plus_epsilon)
    log_phi = log_numerator - log_denominator
    nll = torch.mean(-log_phi)
    mean_pred_var = torch.mean(sigma_hat_plus_epsilon).detach()
    terms = {
        "mean_squared_error": torch.mean(squared_error).detach(),
        "mean_predicted_variance": mean_pred_var,
        "negative_log_likelihood": nll.detach()
    }
    return nll, terms


def compute_loss_on_dataset(model, dataset_loader, loss_function, output_config):
    """
    loss function must have signature (batch_ground_truth: DeviceDict, batch_prediction: DeviceDict) => number, dict_of_labeled_numbers
    """
    with torch.no_grad():
        assert isinstance(model, EchoLearnNN)
        losses = []
        is_implicit = model._output_config.implicit
        res = model._output_config.resolution
        output_fmt = model._output_config.format
        output_dim = model._output_config.dims
        for i, batch in enumerate(dataset_loader):
            if is_implicit:
                batches = make_deterministic_validation_batches_implicit(batch, output_config)
                
                losses_batch = []
                for b in batches:
                    b = b.to(the_device)
                    pred = model(b)
                    loss, _ = loss_function(b, pred)
                    pred = None

                    losses_batch.append(loss.item())
                losses.append(np.mean(np.asarray(losses_batch)))
            else:
                batch = batch.to(the_device)
                pred = model(batch)
                loss, _ = loss_function(batch, pred)
                pred = None
                losses.append(loss.item())
            progress_bar(i, len(dataset_loader))
        return np.mean(np.asarray(losses))
