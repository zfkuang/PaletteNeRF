
import cv2
import numpy as np
import torch
import cv2
import _nbd_palette as _backend

# test function rgb to hsv
if __name__ == "__main__":
    x = torch.rand(100, 3).cuda()
    x_hsv = torch.zeros_like(x) 
    _backend.rgb_to_hsv(x.shape[0], x, x_hsv)
    y = x.cpu().numpy()
    y_hsv = cv2.cvtColor(y[np.newaxis,...], cv2.COLOR_RGB2HSV)[0]
    
    x_recon = torch.zeros_like(x) 
    _backend.hsv_to_rgb(x.shape[0], x_hsv, x_recon)
    y_recon = cv2.cvtColor(y_hsv[np.newaxis,...], cv2.COLOR_HSV2RGB)[0]
    import pdb
    pdb.set_trace()