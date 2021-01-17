import torch

def convert_sdf_to_heatmap(sdf):
    num_dims = len(sdf.shape)
    if num_dims == 4: # B, C, H, W
        return (sdf[:,:1,:,:] <= 0.0).float()
    elif num_dims == 2: # B,
        return (sdf <= 0.0).float()
    else:
        raise Exception("Unrecognized SDF tensor shape")

def convert_heatmap_to_shadowed_heatmap(heatmap):
    num_dims = len(heatmap.shape)
    if num_dims == 4:
        B, C, H, W = heatmap.shape
        mask_row = torch.zeros((B, 1, W))
        out = torch.zeros((B, 1, H, W))
        for i in list(range(H))[::-1]:
            mask_row[heatmap[:,0,i,:] > 0.0] = 1.0
            out[:,0,i,:] = mask_row
        return out
    elif num_dims == 2:
        H, W = heatmap.shape
        mask_row = torch.zeros((W))
        out = torch.zeros((H, W))
        for i in list(range(H))[::-1]:
            mask_row[heatmap[i,:] > 0.0] = 1.0
            out[i] = mask_row
        return out
    else:
        raise Exception("Unrecognized heatmap tensor shape")

def convert_depthmap_to_shadowed_heatmap(depthmap):
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
