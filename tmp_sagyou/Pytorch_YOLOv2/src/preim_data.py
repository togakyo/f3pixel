
import os
import cv2
import time
import numpy as np
import pickle
import torch

from PIL import Image

import matplotlib.pyplot as plt
import config as cfg

import json

def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

def prepare_im_data(img):
    """
    Prepare image data that will be feed to network.

    Arguments:
    img -- PIL.Image object

    Returns:
    im_data -- tensor of shape (3, H, W).
    im_info -- dictionary {height, width}

    """

    im_info = dict()
    im_info['width'], im_info['height'] = (img.shape[1], img.shape[0])

    # resize the image
    H, W = cfg.input_size
    re_img = cv2.resize(img, (H, W))
    im_data = cv2pil(re_img)
    im_npdata = np.asarray(im_data)
    
    print("???im_data.dtype= ",im_npdata.dtype)
    #im_data = np.array(img).resize((H, W))

    # to torch tensor
    #im_data = torch.from_numpy(np.array(im_data)).float() / 255
    im_data = torch.from_numpy(im_npdata).float() / 255

    im_data = im_data.permute(2, 0, 1).unsqueeze(0)

    return im_data, im_info
