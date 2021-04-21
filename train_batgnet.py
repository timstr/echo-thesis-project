import fix_dead_command_line

from Echo4ChDataset import Echo4ChDataset
from BatGNet import BatGNet
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import datetime
import PIL.Image
from the_device import the_device
from custom_collate_fn import custom_collate
from device_dict import DeviceDict
from progress_bar import progress_bar

to_tensor = torchvision.transforms.ToTensor()


def plt_screenshot():
    fig = plt.gcf()
    pil_img = PIL.Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )
    return pil_img
    # return to_tensor(pil_img)


def main():
    parser = ArgumentParser()
    parser.add_argument("--experiment", type=str, dest="experiment", required=True)
    parser.add_argument("--batchsize", type=int, dest="batchsize", default=4)
    parser.add_argument("--nosave", dest="nosave", default=False, action="store_true")
    parser.add_argument("--iterations", type=int, dest="iterations", default=None)
    parser.add_argument("--plotinterval", type=int, dest="plotinterval", default=32)
    parser.add_argument(
        "--validationinterval", type=int, dest="validationinterval", default=256
    )

    args = parser.parse_args()

    if args.nosave:
        print("NOTE: networks are not being saved")

    e4cds = Echo4ChDataset()

    val_ratio = 1 / 32
    val_size = int(len(e4cds) * val_ratio)
    indices_val = list(range(0, val_size))
    indices_train = list(range(val_size, len(e4cds)))

    val_set = torch.utils.data.Subset(e4cds, indices_val)
    train_set = torch.utils.data.Subset(e4cds, indices_train)

    # define the dataset loader (batch size, shuffling, ...)
    collate_fn_device = lambda batch: DeviceDict(custom_collate(batch))

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batchsize,
        num_workers=0,
        pin_memory=False,  # Note, setting pin_memory=False to avoid the pin_memory call
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn_device,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batchsize,
        num_workers=0,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn_device,
    )

    def reconstructionLoss(batch_gt, batch_pred):
        y = batch_gt["output"]
        y_hat = batch_pred["output"]

        assert y.shape[1:] == (64, 64, 64)
        assert y_hat.shape[1:] == (64, 64, 64)

        err = -torch.sum(y * torch.log(y_hat) + (1.0 - y) * torch.log(1 - y_hat))

        # TODO: regularization

        return err, {}

    def validation_loss(model):
        print("Computing validation loss...")
        losses = []
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                batch = batch.to(the_device)
                pred = model(batch)
                loss, _ = reconstructionLoss(batch, pred)
                losses.append(loss.item())
                progress_bar(i, len(val_loader))
            return np.mean(np.asarray(losses))

    network = BatGNet(debug_mode=True).to(the_device)

    def plot_spectrograms(plt_axis, spectrograms):
        assert spectrograms.shape == (8, 256, 256)
        img_grid = torchvision.utils.make_grid(
            spectrograms.unsqueeze(1).repeat(1, 3, 1, 1), nrow=2
        ).permute(1, 2, 0)
        plt_axis.imshow(img_grid)
        plt_axis.axis("off")

    def plot_occupancy_grid(plt_axis, occupancy):
        assert occupancy.shape == (64, 64, 64)
        occupancy = torch.clamp(occupancy, min=0.0, max=1.0)
        img_grid = torchvision.utils.make_grid(
            occupancy.unsqueeze(1).repeat(1, 3, 1, 1), nrow=8
        ).permute(1, 2, 0)
        plt_axis.imshow(img_grid)
        plt_axis.axis("off")

    def save_network(filename):
        filename = "models/" + filename
        print('Saving model to "{}"'.format(filename))
        torch.save(network.state_dict(), filename)

    def restore_network(filename):
        filename = "models/" + filename
        print('Loading model from "{}"'.format(filename))
        network.load_state_dict(torch.load(filename))
        network.eval()

    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)

    timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    log_path = "logs/{}_{}".format(args.experiment, timestamp)

    with SummaryWriter(log_path) as writer:

        plt.ion()

        fig, axes = plt.subplots(2, 4, figsize=(22, 10), dpi=80)
        fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))

        ax_t1 = axes[0, 0]
        ax_t2 = axes[0, 1]
        ax_t3 = axes[0, 2]
        ax_b1 = axes[1, 0]
        ax_b2 = axes[1, 1]
        ax_b3 = axes[1, 2]
        ax_t4 = axes[0, 3]
        ax_b4 = axes[1, 3]

        num_epochs = 1000000
        losses = []
        val_loss_y = []
        val_loss_x = []
        best_val_loss = np.inf
        global_iteration = 0
        for e in range(num_epochs):
            train_iter = iter(train_loader)
            for i in range(len(train_loader)):
                batch_cpu = next(train_iter)
                batch_gpu = batch_cpu.to(the_device)

                pred_gpu = network(batch_gpu)

                loss, loss_terms = reconstructionLoss(batch_gpu, pred_gpu)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

                writer.add_scalar("training loss", loss.item(), global_iteration)
                for k in loss_terms.keys():
                    writer.add_scalar(k, loss_terms[k].item(), global_iteration)

                progress_bar((global_iteration) % args.plotinterval, args.plotinterval)

                if ((global_iteration + 1) % args.validationinterval) == 0:
                    curr_val_loss = validation_loss(network)
                    if curr_val_loss < best_val_loss:
                        best_val_loss = curr_val_loss
                        if not args.nosave:
                            save_network(
                                "{}_{}_model_best.dat".format(
                                    args.experiment, timestamp
                                )
                            )

                    if not args.nosave:
                        save_network(
                            "{}_{}_model_latest.dat".format(args.experiment, timestamp)
                        )

                    val_loss_x.append(len(losses))
                    val_loss_y.append(curr_val_loss)

                    writer.add_scalar(
                        "validation loss", curr_val_loss, global_iteration
                    )

                if ((global_iteration + 1) % args.plotinterval) == 0:
                    pred_cpu = pred_gpu.to("cpu")
                    val_batch_cpu = next(iter(val_loader))
                    val_batch_gpu = val_batch_cpu.to(the_device)

                    # clear figures for a new update
                    ax_t1.cla()
                    ax_b1.cla()
                    ax_t2.cla()
                    ax_b2.cla()
                    ax_t3.cla()
                    ax_b3.cla()
                    ax_t4.cla()
                    ax_b4.cla()

                    # plt.gcf().suptitle(args.experiment)

                    # plot the input
                    ax_t1.title.set_text("Input (train)")
                    plot_spectrograms(ax_t1, batch_cpu["input"][0])
                    ax_b1.title.set_text("Input (validation)")
                    plot_spectrograms(ax_b1, val_batch_cpu["input"][0])

                    # plot the ground truth
                    ax_t2.title.set_text("Ground Truth (train)")
                    plot_occupancy_grid(ax_t2, batch_cpu["output"][0])
                    ax_b2.title.set_text("Ground Truth (validation)")
                    plot_occupancy_grid(ax_b2, val_batch_cpu["output"][0])

                    # plot the predictopm
                    ax_t3.title.set_text("Prediction (train)")
                    plot_occupancy_grid(ax_t3, pred_cpu["output"][0].detach())
                    ax_b3.title.set_text("Prediction (validation)")
                    plot_occupancy_grid(
                        ax_b3, network(val_batch_gpu)["output"][0].detach().cpu()
                    )

                    # plot the training loss on a log plot
                    ax_t4.title.set_text("Training Loss")
                    # ax_t4.set_yscale('log')
                    ax_t4.scatter(range(len(losses)), losses, s=1.0)

                    # plot the validation loss on a log plot
                    ax_b4.title.set_text("Validation Loss")
                    # ax_b4.set_yscale('log')
                    ax_b4.plot(val_loss_x, val_loss_y, c="Red")

                    # Note: calling show or pause will cause a bad time
                    fig.canvas.flush_events()
                    fig.canvas.draw()

                    print(
                        "Epoch {}, iteration {} of {} ({} %), loss={}".format(
                            e,
                            i,
                            len(train_loader),
                            100 * i // len(train_loader),
                            losses[-1],
                        )
                    )

                    if ((global_iteration + 1) % (args.plotinterval * 8)) == 0:
                        plt_screenshot().save(
                            log_path + "/image_" + str(global_iteration + 1) + ".png"
                        )

                        # NOTE: this is done in the screenshot branch so that a screenshot with the final
                        # performance is always included
                        if (
                            args.iterations is not None
                            and global_iteration > args.iterations
                        ):
                            print("Done - desired iteration count was reached")
                            return

                global_iteration += 1

        plt.close("all")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # I don't wanna see any stack traces
        pass
