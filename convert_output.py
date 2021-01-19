from featurize import make_dense_outputs
import torch

def convert_sdf_to_occupancy(sdf):
    num_dims = len(sdf.shape)
    threshold = 0.0 # TODO: tune this (using validation set???)
    if num_dims == 4: # B, C, H, W
        return (sdf[:,:1,:,:] <= threshold).float()
    elif num_dims == 2: # B,
        return (sdf <= threshold).float()
    else:
        raise Exception("Unrecognized SDF tensor shape")

def convert_heatmap_to_occupancy(hm):
    num_dims = len(hm.shape)
    threshold = 0.5 # TODO: tune this (using validation set???)
    if num_dims == 4: # B, C, H, W
        return (hm[:,:1,:,:] >= threshold).float()
    elif num_dims == 2: # B,
        return (hm >= threshold).float()
    else:
        raise Exception("Unrecognized SDF tensor shape")

def convert_occupancy_to_shadowed_occupancy(occupancy):
    num_dims = len(occupancy.shape)
    if num_dims == 4:
        B, C, H, W = occupancy.shape
        mask_row = torch.zeros((B, 1, W))
        out = torch.zeros((B, 1, H, W))
        for i in list(range(H))[::-1]:
            mask_row[occupancy[:,0,i,:] > 0.0] = 1.0
            out[:,0,i,:] = mask_row
        return out
    elif num_dims == 2:
        H, W = occupancy.shape
        mask_row = torch.zeros((W))
        out = torch.zeros((H, W))
        for i in list(range(H))[::-1]:
            mask_row[occupancy[i,:] > 0.0] = 1.0
            out[i] = mask_row
        return out
    else:
        raise Exception("Unrecognized occupancy tensor shape")

def convert_depthmap_to_shadowed_occupancy(depthmap):
    num_dims = len(depthmap.shape)
    if num_dims == 3:
        B, C, W = depthmap.shape
        out = torch.zeros(B, 1, W, W)
        for b in range(B):
            for x in range(W):
                d = int((1.0 - depthmap[b,0,x].item()) * W)
                out[b,0,0:d,x] = 1.0
        return out
    elif num_dims == 1:
        W, = depthmap.shape
        out = torch.zeros(W, W)
        for x in range(W):
            d = int((1.0 - depthmap[x].item()) * W)
            out[0:d,x] = 1.0
        return out
    else:
        raise Exception("Unrecognized depthmap tensor shape")

def ground_truth_occupancy(obstacles, size):
    sdf = make_dense_outputs(obstacles, "sdf", size)
    return convert_sdf_to_occupancy(sdf)

def ground_truth_shadowed_occupancy(obstacles, size):
    o = ground_truth_occupancy(obstacles, size)
    return convert_occupancy_to_shadowed_occupancy(o)