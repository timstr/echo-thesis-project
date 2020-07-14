import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import sys

from wave_kernel import make_wave_kernel, pad_field

def main():
    
    plt.ion()

    size = 64
    
    kernel = make_wave_kernel(
        propagation_speed=1.0,
        velocity_damping=1.0,
        time_step=0.1,
        velocity_dispersion=0.05
    )

    center = 16

    def run_simulation(params, graph_axis, animate=True, clear_screen=True, skip_frames=0, num_steps=800):
        # initial condition
        field = torch.zeros((1, 2, size, size))
        rand_init = torch.rand((2, 2)) * 2.0 - 1.0
        field[:, 0, center-2:center, center-2:center] = rand_init.flip([0, 1])
        field[:, 0, center-2:center, center:center+2] = rand_init.flip([0])
        field[:, 0, center:center+2, center-2:center] = rand_init.flip([1])
        field[:, 0, center:center+2, center:center+2] = rand_init
        field = field.cuda()

        loss = torch.tensor(0.0).cuda()

        graph_axis.cla()

        # Simulation loop
        for i in range(skip_frames + num_steps):
            field = kernel(pad_field(field, params))

            if (animate or not clear_screen) and i % 8 == 0:
                if clear_screen:
                    graph_axis.cla()
                    t = 1.0
                else:
                    t = i / (num_steps - 1)

                graph_axis.imshow(field[0,0,:,:].detach().cpu(), vmin=-0.01, vmax=0.01)

                if animate:
                    plt.show()
                    plt.pause(0.01)

            if (i >= skip_frames):
                topleft = field[:, :, 0:center, 0:center]
                topright = field[:, :, 0:center, center:center*2]
                bottomleft = field[:, :, center:center*2, 0:center]
                bottomright = field[:, :, center:center*2, center:center*2]

                loss += torch.sum((bottomright.flip([2, 3]) - topleft)**2)
                loss += torch.sum((bottomright.flip([2]) - topright)**2)
                loss += torch.sum((bottomright.flip([3]) - bottomleft)**2)
        
        if not animate:
            graph_axis.imshow(field[0,0,:,:].detach().cpu(), vmin=-0.01, vmax=0.01)
            # plt.show()
            # plt.pause(0.05)

        return loss
    
    # depends on:
    # propagation_speed     : VERY YES
    # velocity_damping      : a little bit
    # time_step             : yes
    # velocity_dispersion   : VERY YES

    # Padding parameters for configuration:
    #   propagation_speed = 1.0
    #   velocity_damping = 1.0
    #   time_step = 0.1
    #   velocity_dispersion = 0.05
    # magic_params_normal = torch.Tensor([
    #     0.4125185609, -1.2608314753,  0.4441341460,  0.1328184456,
    #     0.4125185013, -1.2608315945,  0.4441341758,  0.1328190565
    # ])

    # Padding parameters for configuration:
    #   propagation_speed = 0.5 # <- Different
    #   velocity_damping = 1.0
    #   time_step = 0.1
    #   velocity_dispersion = 0.05
    # magic_params_half_propagation_speed = torch.Tensor([
    #     0.3087364137, -0.8797827363,  0.2897737920, -0.4759024978,
    #     0.3087364733, -0.8797824979,  0.2897738814, -0.4759028256
    # ])
    # net diference: [
    #     0.2075642944, -0.381048739,   0.1543603838,  0.6087209434
    # ]

    # Padding parameters for configuration:
    #   propagation_speed = 1.0
    #   velocity_damping = 1.0
    #   time_step = 0.1
    #   velocity_dispersion = 0.1 # <- Different
    # magic_params_double_velocity_dispersion = torch.Tensor([
    #     0.3705597818, -0.6894223094,  0.3786112964, -0.1861230135,
    #     0.3705597222, -0.6894221902,  0.3786113262, -0.1861228794
    # ])
    # net difference: [
    #     0.0419587791, -0.5714091659,  0.0655228496,  0.3189414591
    # ]

    # Padding parameters for configuration:
    #   propagation_speed = 1.0
    #   velocity_damping = 1.0
    #   time_step = 0.05 # <- Different
    #   velocity_dispersion = 0.05
    # magic_params_half_time_step = torch.Tensor([
    #     0.2912884653, -0.2537243068,  0.3076377511, -0.2467312217,
    #     0.2912885547, -0.2537241876,  0.3076378703, -0.2467312515
    # ])
    # net difference: [
    #     0.1212300956, -1.0071071685,  0.1364963949,  0.3795496673
    # ]

    default_params = torch.Tensor([
         0.2731468678,  0.3291435242, -0.5704568028, -0.1916972548, -0.0072381911, -0.0069156736, -0.0045501599,  0.0087358886,
        -0.0653517470,  0.1985650659, -0.3193856180,  0.1051328108, -0.0030024105, -0.0037237250, -0.0020352036,  0.0133369453,
         0.2357937992,  0.2602138519, -0.5818488598, -0.2399103791, -0.0069989869, -0.0064358339, -0.0043960437,  0.0090550631,
        -0.1118448377,  0.1101172492, -0.3644709289,  0.0389745384, -0.0027823041, -0.0032603526, -0.0019526740,  0.0135646062
    ]).cuda()
    
    # default_params = torch.zeros(32).cuda()
    params = torch.nn.Parameter(default_params, requires_grad=True)

    # run_simulation(default_params, plt.subplots(1, 1)[1], num_steps=512)
    # return
    ####################################

    # TODO: experiment with random parameter initialization and see how this affects converged values

    optimizer = torch.optim.Adam([params], lr=0.001)

    torch.set_printoptions(precision=10)

    fig, ax = plt.subplots(1, 2)

    losses = []
    for i in range(1024):
        sys.stdout.write("Running simulation...")
        loss = run_simulation(params, graph_axis=ax[0], animate=True, clear_screen=True, skip_frames=120, num_steps=150)
        sys.stdout.write(" done\n")
        optimizer.zero_grad()
        sys.stdout.write("Computing gradient...")
        loss.backward(retain_graph=True)
        sys.stdout.write(" done\n")
        optimizer.step()
        losses.append(loss.item())

        print(i)
        print("Loss: ", loss.item())
        print("Parameters: ", params.detach())

        ax[1].cla()
        ax[1].scatter(range(len(losses)), losses, s=1.0)
        plt.show()
        plt.pause(0.05)



if __name__ == "__main__":
    main()