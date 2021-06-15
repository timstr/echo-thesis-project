import PIL
import numpy as np


def plt_screenshot(plt_figure):
    pil_img = PIL.Image.frombytes(
        "RGB", plt_figure.canvas.get_width_height(), plt_figure.canvas.tostring_rgb()
    )
    return pil_img


class LossPlotter:
    def __init__(self, aggregation_interval, colour, num_quantiles):
        assert isinstance(aggregation_interval, int)
        r, g, b = colour
        assert isinstance(r, float)
        assert isinstance(g, float)
        assert isinstance(b, float)
        assert isinstance(num_quantiles, int)
        assert num_quantiles >= 1 and num_quantiles < aggregation_interval
        self._colour_r = r
        self._colour_g = g
        self._colour_b = b
        self._aggregation_interval = aggregation_interval
        self._colour = colour
        self._num_quantiles = num_quantiles
        self._values = [[] for _ in range(num_quantiles + 1)]
        self._acc = []

    def append(self, item):
        assert isinstance(item, float)
        self._acc.append(item)
        if len(self._acc) == self._aggregation_interval:
            q = np.linspace(0.0, 1.0, num=(self._num_quantiles + 1))
            qv = np.quantile(self._acc, q)
            for i in range(self._num_quantiles + 1):
                self._values[i].append(qv[i])
            self._acc = []

    def plot_to(self, plt_axis):

        if self._aggregation_interval > 1:
            x_min = self._aggregation_interval // 2
            x_stride = self._aggregation_interval
            x_count = len(self._values[0])
            x_values = range(x_min, x_min + x_count * x_stride, x_stride)
            for i in range(self._num_quantiles):
                t = i / self._num_quantiles
                t = 2.0 * min(t, 1.0 - t)
                c = (self._colour_r, self._colour_g, self._colour_b, t)
                plt_axis.fill_between(
                    x=x_values, y1=self._values[i], y2=self._values[i + 1], color=c
                )
        else:
            plt_axis.scatter(
                range(self._values[0]),
                self._items,
                s=1.0,
                color=(self._colour_r, self._colour_g, self._colour_b),
            )
