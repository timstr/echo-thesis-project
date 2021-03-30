from featurize_audio import crop_audio_from_location
from question_mark import question_marks
import torch
import torchvision
import PIL.Image

from device_dict import DeviceDict
from config import InputConfig, OutputConfig
from EchoLearnNN import EchoLearnNN
from featurize import make_dense_implicit_output_pred, make_dense_tof_cropped_output_pred, make_depthmap_gt, make_echo4ch_dense_implicit_output_pred, make_heatmap_image_gt, make_sdf_image_gt, purple_yellow, colourize_sdf, blue_yellow

def plot_inputs(plt_axis, batch, input_config):
    assert isinstance(batch, DeviceDict)
    assert isinstance(input_config, InputConfig)
    with torch.no_grad():
        the_input = batch['input'][0].detach()
        if input_config.tof_cropping:
            plt_axis.set_ylim(0.0, 1.0)
            plt_axis.axis("off")
            # HACK
            location_y = 0.5
            location_x = 0.5
            the_input_with_fake_feature_dim = the_input.unsqueeze(1)
            the_input_cropped = crop_audio_from_location(the_input_with_fake_feature_dim, input_config, location_y, location_x)
            the_input_cropped = the_input_cropped.squeeze(1)
            for j in range(input_config.num_channels):
                top = j / input_config.num_channels
                bottom = (j + 1) / input_config.num_channels
                center = (top + bottom) * 0.5
                scale = bottom - top
                plt_axis.plot(center + scale * the_input_cropped[j].detach())
        elif (len(the_input.shape) == 2):
            plt_axis.set_ylim(-1, 1)
            for j in range(input_config.num_channels):
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

def plot_dense_output_echo4ch(plt_axis, batch, output_config):
    assert isinstance(batch, DeviceDict)
    assert isinstance(output_config, OutputConfig)
    assert output_config.using_echo4ch
    with torch.no_grad():
        if output_config.format == "depthmap":
            gt = batch["gt_depthmap"][0].detach().cpu()
            assert gt.shape == (64, 64)
            gt = 1.0 - gt # To invert colours
            plt_axis.imshow(purple_yellow(gt), interpolation="bicubic")
            plt_axis.axis("off")
        elif output_config.format == "heatmap":
            gt = batch["gt_heatmap"][0].detach().cpu()
            assert gt.shape == (64, 64, 64)
            img_grid = torchvision.utils.make_grid(
                gt.unsqueeze(1).repeat(1, 3, 1, 1),
                nrow=8,
                pad_value=0.25
            )
            img_grid_colourized = purple_yellow(img_grid[0])
            plt_axis.imshow(img_grid_colourized)
            plt_axis.axis("off")
        else:
            raise Exception("Unrecognized output representation")

def show_sampling_location(plt_axis, output_config):
    # HACK
    return

    assert isinstance(output_config, OutputConfig)
    # HACK
    location_y = output_config.resolution * 0.5
    location_x = output_config.resolution * 0.5

    size = output_config.resolution * 0.02
    thickness = 2.0
    plt_axis.plot([location_x - size, location_x + size], [location_y - size, location_y + size], c="red", linewidth=thickness)
    plt_axis.plot([location_x - size, location_x + size], [location_y + size, location_y - size], c="red", linewidth=thickness)

def plot_ground_truth(plt_axis, batch, output_config, show_samples=False):
    assert isinstance(batch, DeviceDict)
    assert isinstance(output_config, OutputConfig)
    with torch.no_grad():
        if output_config.using_echo4ch:
            plot_dense_output_echo4ch(plt_axis, batch, output_config)
            return
        if output_config.format == "sdf":
            img = make_sdf_image_gt(batch, output_config.resolution).cpu()
            plt_axis.imshow(colourize_sdf(img), interpolation="bicubic")
            plt_axis.axis("off")
            if show_samples and output_config.implicit:
                yx = batch["params"][0].detach() * output_config.resolution
                plt_axis.scatter(yx[:,1], yx[:,0], s=1.0)
            if output_config.tof_cropping:
                show_sampling_location(plt_axis, output_config)
        elif output_config.format == "heatmap":
            img = make_heatmap_image_gt(batch, output_config.resolution).cpu()
            plt_axis.imshow(blue_yellow(img), interpolation="bicubic")
            plt_axis.axis("off")
            if show_samples and output_config.implicit:
                yx = batch["params"][0].detach() * output_config.resolution
                plt_axis.scatter(yx[:,1], yx[:,0], s=1.0)
            if output_config.tof_cropping:
                show_sampling_location(plt_axis, output_config)
        elif output_config.format == "depthmap":
            arr = make_depthmap_gt(batch, output_config.resolution).cpu()
            plt_axis.set_ylim(-0.5, 1.5)
            plt_axis.plot(arr)
            # TODO: show samples somehow (maybe by dots along gt arr)
        else:
            raise Exception("Unrecognized output representation")

def plot_image(plt_axis, img, display_fn, output_config, checkerboard_size=8):
    assert isinstance(output_config, OutputConfig)
    C, H, W = img.shape
    assert H == W
    N = H
    with torch.no_grad():
        y = img[0]
        plt_axis.imshow(display_fn(y), interpolation="bicubic")
        if (output_config.predict_variance):
            sigma = img[1]
            sigma_clamped = torch.clamp(sigma, 0.0, 1.0)
            gamma_value = 0.5
            sigma_curved = sigma_clamped**gamma_value
            
            # checkerboard_res = N // checkerboard_size
            # ls = torch.linspace(0.0, (checkerboard_res - 1), N)
            # square_wave = 2.0 * torch.round(0.5 * ls - torch.floor(0.5 * ls)) - 1.0
            # checkerboard_y, checkerboard_x = torch.meshgrid(square_wave, -square_wave)
            # checkerboard = 0.5 + 0.5 * checkerboard_y * checkerboard_x

            # sigma_curved *= checkerboard

            sigma_curved *= (1.0 - question_marks(H, W, 18, True))

            mask = torch.cat((
                torch.zeros((N, N, 3)), # RGB
                sigma_curved.unsqueeze(-1), # Alpha
            ), dim=2)
            plt_axis.imshow(mask, interpolation="bicubic")
        plt_axis.axis("off")

def plot_depthmap(plt_axis, data, output_config):
    assert isinstance(output_config, OutputConfig)
    assert output_config.format == "depthmap"
    with torch.no_grad():
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
    with torch.no_grad():
        if output_config.using_echo4ch:
            if output_config.implicit:
                output = make_echo4ch_dense_implicit_output_pred(batch, network, output_config)
            else:
                output = network(batch)['output'][0]
            if output_config.format == "depthmap":
                output[0] = 1.0 - output[0] # To invert colours (purple is far, yellow (object) is near)
            elif output_config.format == "heatmap":
                assert output.shape == (output_config.num_channels, 64, 64, 64)
                output_as_minibatch = output.permute(1, 0, 2, 3)
                output_grid = torchvision.utils.make_grid(
                    output_as_minibatch[:,0:1].repeat(1, 3, 1, 1),
                    nrow=8,
                    pad_value=0.25
                )[:1]
                if output_config.predict_variance:
                    output_variance_grid = torchvision.utils.make_grid(
                        output_as_minibatch[:,1:2].repeat(1, 3, 1, 1),
                        nrow=8
                    )[:1]
                    output_grid = torch.cat((
                        output_grid,
                        output_variance_grid
                    ), dim=0)
                output = output_grid
                assert len(output.shape) == 3
            output = output.cpu().detach()
            checkerboard_size = 2 if (output_config.format == "depthmap") else 8
            plot_image(plt_axis, output, purple_yellow, output_config, checkerboard_size=checkerboard_size)
            return

        if output_config.tof_cropping:
            output = make_dense_tof_cropped_output_pred(batch, network, output_config)
        elif output_config.implicit:
            output = make_dense_implicit_output_pred(batch, network, output_config)
        else:
            output = network(batch)['output'][0].detach().cpu()

        if output_config.format == "sdf":
            plot_image(plt_axis, output, colourize_sdf, output_config)
        elif output_config.format == "heatmap":
            plot_image(plt_axis, output, blue_yellow, output_config)
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