import fix_dead_command_line
from featurize import all_possible_obstacles
import matplotlib.pyplot as plt

from wave_field import Field
from utils import progress_bar


def main():
    plt.ion()

    all_of_them = list(
        all_possible_obstacles(2, 0.05, 0.1, 0.3, 0.1, 0.8, 0.1, 0.9, 3, 4, 4)
    )

    for i, o in enumerate(all_of_them):
        f = Field(512)
        f.add_obstacles(o)
        plt.cla()
        plt.imshow(f.to_image().cpu())
        plt.gcf().canvas.draw()
        plt.gcf().canvas.flush_events()
        progress_bar(i, len(all_of_them))


if __name__ == "__main__":
    main()
