import numpy as np
import math

from field import Field, make_random_field
from raymarch import raymarch
import matplotlib.pyplot as plt

def main():
    size = 512
    field = make_random_field(size, size)

    plt.imshow(field.get_barrier()[0,0,:,:].cpu().numpy())
    plt.show()

    fov = 360
    res = 128
    depthmap = raymarch(
        field,
        size//2,
        size//2,
        0,
        1,
        fov=fov,
        res=res,
        step_size=1
    )
    degrees = np.linspace(0, fov, res)
    
    plt.plot(depthmap)
    plt.show()
    plt.polar(degrees * math.pi / 180.0, depthmap)
    plt.show()

if __name__ == "__main__":
    main()
