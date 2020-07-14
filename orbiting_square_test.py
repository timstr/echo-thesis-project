import torch
from argparse import ArgumentParser
import matplotlib.animation
import matplotlib.pyplot as plt

from dataset import WaveSimDataset
from device_dict import DeviceDict
from custom_collate_fn import custom_collate
from progress_bar import progress_bar
from featurize import make_sdf_image_gt, make_sdf_image_pred
from ObstacleSDFNet import ObstacleSDFNet

def load_model(path):
    filename = "models/" + path
    network = ObstacleSDFNet().cuda()
    print("Loading model from \"{}\"".format(filename))
    network.load_state_dict(torch.load(filename))
    network.eval()
    return network

def render_test_animation(network):
    ds = WaveSimDataset(data_folder="dataset/v7_test", permute=False, samples_per_example=1)
    collate_fn_device = lambda batch : DeviceDict(custom_collate(batch))
    ld = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        num_workers=0,
        pin_memory=False, # Note, setting pin_memory=False to avoid the pin_memory call
        shuffle=False,
        drop_last=True,
        collate_fn=collate_fn_device
    )
    plt.axis('off')
    # fig = plt.figure(
    #     figsize=(8,8),
    #     dpi=64
    # )
    it = iter(ld)
    def next_data():
        batch = next(it).to('cuda')
        return make_sdf_image_pred(batch, 32, network)
        # return make_sdf_image_gt(batch, 32)
    
    num_frames = len(ld)
    def animate(i):
        plt.clf()
        plt.imshow(next_data(), vmin=0.0, vmax=0.5, cmap='hsv')
        progress_bar(i, num_frames)
    output_path = "test set sdf prediction.mp4"
    fps = 30
    ani = matplotlib.animation.FuncAnimation(
        plt.gcf(),
        animate,
        frames=num_frames,
        interval = 1000/fps
    )

    ani.save(output_path)

    sys.stdout.write("\nSaved animation to {}".format(output_path))

if __name__ == "__main__":
    render_test_animation(load_model("sdf_from_conv_summary_stats_07-07-2020_12-20-44_model_latest.dat"))
