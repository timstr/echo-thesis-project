import numpy as np
import torch
import torch.nn
import matplotlib.pyplot as plt
import math
import sys
import pickle

from the_device import the_device
from wave_simulation import step_simulation

def make_field_image(field):
    h, w = field.shape
    amp = 5.0
    field_rgb = torch.zeros(h, w, 3)
    # Red: positive amplitude
    field_rgb[:,:,0] = torch.clamp(amp * field, 0.0, 1.0)
    # Blue: negative amplitude
    field_rgb[:,:,2] = torch.clamp(-amp * field , 0.0, 1.0)
    # Green: absolute amplitude minus one
    field_rgb[:,:,1] = torch.clamp(torch.abs(-amp * field) - 1.0, 0.0, 1.0)
    return field_rgb

def main():
    
    plt.ion()

    size = 64
    center = 16

    def run_simulation(half_pad_kernel, graph_axis, animate=True, num_batches=1, skip_frames=0, num_steps=800):

        loss = torch.tensor(0.0, dtype=torch.float).to(the_device)

        for b in range(num_batches):
            # initial condition
            field_now = torch.zeros(size, size).to(the_device)
            field_prev = torch.zeros(size, size).to(the_device)
            rad = 6
            res = 3
            rand_init = torch.rand(rad//res, rad//res).repeat_interleave(res, dim=0).repeat_interleave(res, dim=1) * 2.0 - 1.0
            field_now[center-rad:center, center-rad:center] = rand_init.flip([0, 1])
            field_now[center-rad:center, center:center+rad] = rand_init.flip([0])
            field_now[center:center+rad, center-rad:center] = rand_init.flip([1])
            field_now[center:center+rad, center:center+rad] = rand_init

            # Simulation loop
            for i in range(skip_frames + num_steps):
                field_now, field_prev = step_simulation(field_now, field_prev, half_pad_kernel)

                if b == 0 and animate:# and i % 8 == 0:
                    if animate:
                        graph_axis.cla()
                        graph_axis.imshow(make_field_image(field_now).detach().cpu())
                        plt.gcf().canvas.draw()
                        plt.gcf().canvas.flush_events()

                if (i >= skip_frames):
                    topleft = field_now[0:center, 0:center]
                    topright = field_now[0:center, center:center*2]
                    bottomleft = field_now[center:center*2, 0:center]
                    bottomright = field_now[center:center*2, center:center*2]

                    loss += torch.mean((bottomright.flip([0, 1]) - topleft)**2)
                    loss += torch.mean((bottomright.flip([0]) - topright)**2)
                    loss += torch.mean((bottomright.flip([1]) - bottomleft)**2)
                    
            if b == 0 and not animate:
                graph_axis.cla()
                graph_axis.imshow(make_field_image(field_now).detach().cpu())
                # plt.gcf().canvas.draw()
                # plt.gcf().canvas.flush_events()

        return loss


    # half_pad_kernel = torch.zeros(4, 4, 2, 2).to(the_device)

    with open("sim_params/current/v7/sim_params_latest.pkl", "rb") as f:
        params = pickle.load(f)
        half_pad_kernel = torch.tensor(params["half_pad_kernel"], dtype=torch.float).to(the_device)
        # half_pad_kernel = torch.cat((
        #     half_pad_kernel,
        #     torch.zeros(2, 2, 2, 2).to(the_device)
        # ), dim=1)

    half_pad_kernel = torch.nn.Parameter(half_pad_kernel, requires_grad=True)

    optimizer = torch.optim.Adam([half_pad_kernel], lr=0.0001)

    torch.set_printoptions(precision=10)

    fig, ax = plt.subplots(1, 2)

    losses = []

    best_loss = 1e6

    for i in range(1024*1024):
        loss = run_simulation(
            half_pad_kernel,
            graph_axis=ax[0],
            animate=(i % 8 == 0),
            num_batches=8,
            skip_frames=0,
            num_steps=48
        )
        optimizer.zero_grad()
        if math.isnan(loss.item()):
            print("The loss was NaN :(")
            continue
        loss.backward(retain_graph=True)
        optimizer.step()
        losses.append(loss.item())

        print(i)
        print("Loss: ", loss.item())

        def dump(fname):
            print(f"Saving parameters to {fname}...")
            with open(fname, "wb") as f:
                pickle.dump({
                    "half_pad_kernel": half_pad_kernel.detach().cpu().numpy(),
                }, f)

        if i % 10 == 0:
            dump("sim_params_latest.pkl")

        if loss.item() < best_loss:
            best_loss = loss.item()
            dump("sim_params_best.pkl")

        ax[1].cla()
        ax[1].scatter(range(len(losses)), losses, s=1.0, c="blue")
        ax[1].set_yscale('log')
        plt.gcf().canvas.draw()
        plt.gcf().canvas.flush_events()



if __name__ == "__main__":
    main()