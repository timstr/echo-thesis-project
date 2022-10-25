import fix_dead_command_line
import cleanup_when_killed

import random
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import os
import datetime
from argparse import ArgumentParser

from batgnet import BatGNet

from assert_eq import assert_eq
from signals_and_geometry import backfill_occupancy
from torch_utils import restore_module, save_module
from which_device import get_compute_device
from device_dict import DeviceDict
from utils import progress_bar
from torch.utils.data._utils.collate import default_collate
from Echo4ChDatasetH5 import Echo4ChDataset, k_spectrograms, k_occupancy
from dataset_adapters import occupancy_grid_to_depthmap


def uint8_to_float(x):
    assert isinstance(x, torch.Tensor)
    assert_eq(x.dtype, torch.uint8)
    return x.float() / 255


def bool_to_float(x):
    assert isinstance(x, torch.Tensor)
    assert_eq(x.dtype, torch.bool)
    return x.float()


def concat_images(img1, img2, *rest, horizontal=True):
    imgs = [img1, img2]
    imgs.extend(rest)
    imgs = [
        torch.clamp(
            img.unsqueeze(0).repeat(3, 1, 1) if (img.ndim == 2) else img,
            min=0.0,
            max=1.0,
        )
        for img in imgs
    ]
    return torchvision.utils.make_grid(imgs, nrow=(len(imgs) if horizontal else 1))


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--experiment",
        type=str,
        dest="experiment",
        required=True,
        help="short description or mnemonic of reason for training, used in log files and model names",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        dest="batchsize",
        default=4,
        help="batch size used for each training iteration",
    )
    parser.add_argument(
        "--learningrate",
        type=float,
        dest="learningrate",
        default=1e-4,
        help="Adam optimizer learning rate",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        dest="adam_beta1",
        default=0.9,
        help="Adam optimizer parameter beta 1",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        dest="adam_beta2",
        default=0.999,
        help="Adam optimizer parameter beta 2",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        dest="iterations",
        default=None,
        help="if specified, number of iterations to train for. Trains forever if not specified.",
    )
    parser.add_argument(
        "--nosave",
        dest="nosave",
        default=False,
        action="store_true",
        help="do not save model files",
    )
    parser.add_argument(
        "--plotinterval",
        type=int,
        dest="plotinterval",
        default=512,
        help="number of training iterations between generating visualizations",
    )
    parser.add_argument(
        "--restoremodelpath", type=str, dest="restoremodelpath", default=None
    )
    parser.add_argument(
        "--restoreoptimizerpath", type=str, dest="restoreoptimizerpath", default=None
    )

    args = parser.parse_args()

    assert (args.restoremodelpath is None) == (args.restoreoptimizerpath is None)

    k_env_dataset_train = "ECHO4CH_DATASET_TRAIN"

    dataset_train_path = os.environ.get(k_env_dataset_train)
    if dataset_train_path is None or not os.path.isfile(dataset_train_path):
        raise Exception(
            f"Please set the {k_env_dataset_train} environment variable to point to the ECHO4CH dataset HDF5 file for training"
        )

    dataset_train = Echo4ChDataset(path_to_h5file=dataset_train_path)

    def collate_fn_device(batch):
        return DeviceDict(default_collate(batch))

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batchsize,
        num_workers=0,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn_device,
    )

    def plot_images(the_model):
        visualization_begin_time = datetime.datetime.now()
        print("Generating visualizations...")

        example_train = random.choice(dataset_train).to(get_compute_device())

        threshold = 0.5

        def make_image(the_example):
            spectrograms = uint8_to_float(the_example[k_spectrograms])
            occupancy_gt = backfill_occupancy(the_example[k_occupancy])
            assert_eq(occupancy_gt.shape, (64, 64, 64))
            occupancy_pred = the_model(spectrograms.unsqueeze(0)).squeeze(0)
            assert_eq(occupancy_pred.shape, (64, 64, 64))
            occupancy_pred_binary = occupancy_pred >= threshold
            depthmaps_gt = concat_images(
                occupancy_grid_to_depthmap(occupancy_gt.flip(0), 0).permute(1, 0),
                occupancy_grid_to_depthmap(occupancy_gt, 1).permute(1, 0),
                occupancy_grid_to_depthmap(occupancy_gt, 2).permute(1, 0),
                horizontal=False,
            )
            depthmaps_pred = concat_images(
                occupancy_grid_to_depthmap(occupancy_pred_binary.flip(0), 0).permute(
                    1, 0
                ),
                occupancy_grid_to_depthmap(occupancy_pred_binary, 1).permute(1, 0),
                occupancy_grid_to_depthmap(occupancy_pred_binary, 2).permute(1, 0),
                horizontal=False,
            )
            projections_pred = concat_images(
                torch.mean(occupancy_pred, dim=0).permute(1, 0),
                torch.mean(occupancy_pred, dim=1).permute(1, 0),
                torch.mean(occupancy_pred, dim=2).permute(1, 0),
                horizontal=False,
            )
            slices_pred = concat_images(
                occupancy_pred[occupancy_pred.shape[0] // 2, :, :].permute(1, 0),
                occupancy_pred[:, occupancy_pred.shape[1] // 2, :].permute(1, 0),
                occupancy_pred[:, :, occupancy_pred.shape[2] // 2].permute(1, 0),
                horizontal=False,
            )

            return concat_images(
                depthmaps_gt, depthmaps_pred, projections_pred, slices_pred
            )

        writer.add_image(
            "Depthmaps Ground Truth, Depthmap Prediction, Projections Prediction, Slices Prediction (train)",
            make_image(example_train),
            global_iteration,
        )

        visualization_end_time = datetime.datetime.now()
        duration = visualization_end_time - visualization_begin_time
        seconds = float(duration.seconds) + (duration.microseconds / 1_000_000.0)
        print(f"Generating visualizations done after {seconds} seconds.")

    model = BatGNet()

    model = model.to(get_compute_device())

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learningrate,
        betas=(args.adam_beta1, args.adam_beta2),
    )

    if args.restoremodelpath is not None:
        restore_module(model, args.restoremodelpath)
        restore_module(optimizer, args.restoreoptimizerpath)

    timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    if not args.nosave:
        base_model_path = os.environ.get("TRAINING_MODEL_PATH")

        if base_model_path is None:
            raise Exception(
                "Please set the TRAINING_MODEL_PATH environment variable to point to the desired model directory"
            )
        if not os.path.exists(base_model_path):
            os.makedirs(base_model_path)

        model_path = os.path.join(base_model_path, f"{args.experiment}_{timestamp}")
        if os.path.exists(model_path):
            raise Exception(
                f'Error: attempted to create a folder for saving models at "{model_path}" but the folder already exists.'
            )

        os.makedirs(model_path)

    log_path_root = os.environ.get("TRAINING_LOG_PATH")

    if log_path_root is None:
        raise Exception(
            "Please set the TRAINING_LOG_PATH environment variable to point to the desired log directory"
        )

    if not os.path.exists(log_path_root):
        os.makedirs(log_path_root)

    log_folder_name = f"{args.experiment}_{timestamp}"

    log_path = os.path.join(log_path_root, log_folder_name)

    def save_things(suffix=None):
        assert suffix is None or isinstance(suffix, str)
        if args.nosave:
            return
        save_module(
            model,
            os.path.join(
                model_path,
                f"model_batgnet_echo4ch_{suffix or ''}.dat",
            ),
        )
        save_module(
            optimizer,
            os.path.join(
                model_path,
                f"optimizer_batgnet_echo4ch_{'' if suffix is None else ('_' + suffix)}.dat",
            ),
        )

    global_iteration = 0

    try:
        with SummaryWriter(log_path) as writer:

            num_epochs = 1000000

            for i_epoch in range(num_epochs):
                train_iter = iter(train_loader)
                for i_example in range(len(train_loader)):
                    example_batch = next(train_iter).to(get_compute_device())

                    inputs = uint8_to_float(example_batch[k_spectrograms])
                    gt = bool_to_float(backfill_occupancy(example_batch[k_occupancy]))
                    pred = model(inputs)
                    assert_eq(gt.shape, pred.shape)
                    loss = torch.mean(torch.square(pred - gt))

                    optimizer.zero_grad()

                    loss.backward()

                    optimizer.step()

                    loss = loss.item()

                    writer.add_scalar("training loss", loss, global_iteration)

                    progress_bar(
                        (global_iteration) % args.plotinterval, args.plotinterval
                    )

                    time_to_plot = (
                        (global_iteration + 1) % args.plotinterval
                    ) == 0 or (global_iteration == 0)

                    if time_to_plot:
                        model.eval()

                        print(
                            f"Epoch {i_epoch}, {global_iteration + 1} total iterations"
                        )

                        plot_images(model)

                        if (
                            args.iterations is not None
                            and global_iteration > args.iterations
                        ):
                            print("Done - desired iteration count was reached")
                            return

                        model.train()

                    global_iteration += 1

    except KeyboardInterrupt:
        if args.nosave:
            print(
                "\n\nControl-C detected, but not saving model due to --nosave option\n"
            )
        else:
            print("\n\nControl-C detected, saving model...\n")
            save_things("aborted")
            print("Exiting")
        exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # I don't wanna see any stack traces
        pass
