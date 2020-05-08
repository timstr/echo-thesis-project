import pickle
import numpy as np

from field import Field, make_random_field
from raymarch import raymarch

def main():
    field_size = 512
    fov = 360
    res = 128
    sim_len = 8192

    receiver_locations = [
        (field_size//2 - 10, field_size//2 - 10),
        (field_size//2 - 10, field_size//2 + 10),
        (field_size//2 + 10, field_size//2 - 10),
        (field_size//2 + 10, field_size//2 + 10),
    ]

    # returns list_of_obstacles, depth_map, sound_buffer
    def make_example():
        print("Creating field")
        field = make_random_field(field_size, field_size)
        print("Raymarching")
        depthmap = raymarch(
            field,
            field_size//2,
            field_size//2,
            dirx=1,
            diry=0,
            fov=fov,
            res=res,
            step_size=1
        )
        print("Simulating waves")
        field.get_field()[0,0,field_size//2, field_size//2] = 10.0
        receiver_buf = []
        for i in range(sim_len):
            v = []
            for y, x in receiver_locations:
                ff = field.get_field()
                amp = ff[0,0,y,x].item()
                v.append(amp)
            receiver_buf.append(v)
            field.step()
        
        sound_buf = np.asarray(receiver_buf)
        sound_max_amp = np.max(np.abs(sound_buf))
        if sound_max_amp > 1e-3:
            sound_buf *= 0.5 / sound_max_amp
        print("Done")
        return field.get_obstacles(), depthmap, sound_buf


    dataset_size = 8000
    output_path = "dataset/test"

    for i in range(dataset_size):
        print("Creating example ", i)
        example = make_example()
        fname = "{0}/example {1}.pkl".format(output_path, i)
        with open(fname, "wb") as outfile:
            pickle.dump(example, outfile)

if __name__ == "__main__":
    main()