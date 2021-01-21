import torch
import torchvision
import PIL.Image

from device_dict import DeviceDict
from dataset_config import OutputConfig, ReceiverConfig
from EchoLearnNN import EchoLearnNN
from the_device import what_my_gpu_can_handle
from featurize import make_depthmap_gt, make_depthmap_pred, make_heatmap_image_gt, make_heatmap_image_pred, make_sdf_image_gt, make_sdf_image_pred, red_white_blue, red_white_blue_banded

def plot_inputs(plt_axis, batch, receiver_config):
    assert isinstance(batch, DeviceDict)
    assert isinstance(receiver_config, ReceiverConfig)
    the_input = batch['input'][0].detach()
    if (len(the_input.shape) == 2):
        plt_axis.set_ylim(-1, 1)
        for j in range(receiver_config.count):
            plt_axis.plot(the_input[j].detach())
    else:
        the_input_min = torch.min(the_input)
        the_input_max = torch.max(the_input)
        the_input = (the_input - the_input_min) / (the_input_max - the_input_min)
        spectrogram_img_grid = torchvision.utils.make_grid(
            the_input.unsqueeze(1).repeat(1, 3, 1, 1),
            nrow=1
        )
        plt_axis.imshow(spectrogram_img_grid.permute(1, 2, 0))
        plt_axis.axis("off")

def plot_ground_truth(plt_axis, batch, output_config, show_samples=False):
    assert isinstance(batch, DeviceDict)
    assert isinstance(output_config, OutputConfig)
    if output_config.format == "sdf":
        img = make_sdf_image_gt(batch, output_config.resolution).cpu()
        plt_axis.imshow(red_white_blue_banded(img), interpolation="bicubic")
        plt_axis.axis("off")
        if show_samples and output_config.implicit:
            yx = batch["params"][0].detach() * output_config.resolution
            plt_axis.scatter(yx[:,1], yx[:,0], s=1.0)
    elif output_config.format == "heatmap":
        img = make_heatmap_image_gt(batch, output_config.resolution).cpu()
        plt_axis.imshow(red_white_blue(img), interpolation="bicubic")
        plt_axis.axis("off")
        if show_samples and output_config.implicit:
            yx = batch["params"][0].detach() * output_config.resolution
            plt_axis.scatter(yx[:,1], yx[:,0], s=1.0)
    elif output_config.format == "depthmap":
        arr = make_depthmap_gt(batch, output_config.resolution).cpu()
        plt_axis.set_ylim(-0.5, 1.5)
        plt_axis.plot(arr)
        # TODO: show samples somehow (maybe by dots along gt arr)
    else:
        raise Exception("Unrecognized output representation")

def plot_image(plt_axis, img, display_fn, output_config):
    assert isinstance(output_config, OutputConfig)
    y = img[0]
    assert y.shape == (output_config.resolution, output_config.resolution)
    plt_axis.imshow(display_fn(y), interpolation="bicubic")
    if (output_config.predict_variance):
        sigma = img[1]
        assert sigma.shape == (output_config.resolution, output_config.resolution)
        sigma_clamped = torch.clamp(sigma, 0.0, 1.0)
        gamma_value = 0.5
        sigma_curved = sigma_clamped**gamma_value
        mask = torch.cat((
            torch.zeros((output_config.resolution, output_config.resolution, 3)),
            sigma_curved.unsqueeze(-1),
        ), dim=2)
        plt_axis.imshow(mask, interpolation="bicubic")
    plt_axis.axis("off")

def plot_depthmap(plt_axis, data, output_config):
    assert isinstance(output_config, OutputConfig)
    assert output_config.format == "depthmap"
    y = data[0]
    assert y.shape == (output_config.resolution,)
    plt_axis.set_ylim(-0.5, 1.5)
    plt_axis.plot(y, c="black")
    if (output_config.predict_variance):
        sigma = data[1]
        assert sigma.shape == (output_config.resolution,)
        plt_axis.plot(y - sigma, c="red")
        plt_axis.plot(y + sigma, c="red")

def plot_prediction(plt_axis, batch, network, output_config):
    assert isinstance(batch, DeviceDict)
    assert isinstance(network, EchoLearnNN)
    assert isinstance(output_config, OutputConfig)
    if output_config.implicit:
        if output_config.format == "sdf":
            num_splits = output_config.resolution**2 // what_my_gpu_can_handle
            img = make_sdf_image_pred(batch, output_config.resolution, network, num_splits, output_config.predict_variance)
            plot_image(plt_axis, img, red_white_blue_banded, output_config)
        elif output_config.format == "heatmap":
            num_splits = output_config.resolution**2 // what_my_gpu_can_handle
            img = make_heatmap_image_pred(batch, output_config.resolution, network, num_splits, output_config.predict_variance)
            plot_image(plt_axis, img, red_white_blue, output_config)
        elif output_config.format == "depthmap":
            arr = make_depthmap_pred(batch, output_config.resolution, network)
            plot_depthmap(plt_axis, arr, output_config)
        else:
            raise Exception("Unrecognized output representation")
    else:
        # non-implicit function
        output = network(batch)['output'][0].detach().cpu()
        if output_config.format == "sdf":
            plot_image(plt_axis, output, red_white_blue_banded, output_config)
        elif output_config.format == "heatmap":
            plot_image(plt_axis, output, red_white_blue, output_config)
        elif output_config.format == "depthmap":
            plot_depthmap(plt_axis, output, output_config)
        else:
            raise Exception("Unrecognized output representation")


def plt_screenshot(plt_figure):
    pil_img = PIL.Image.frombytes(
        'RGB',
        plt_figure.canvas.get_width_height(),
        plt_figure.canvas.tostring_rgb()
    )
    return pil_img