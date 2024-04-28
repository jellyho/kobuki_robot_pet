import numpy as np
import torch

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5

    #tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
    array = np.array([brg_img.transpose(2, 0, 1)])
    tensor = torch.tensor(array).float()
    return tensor