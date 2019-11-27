import os
import glob
import re

import numpy as np
import torch

from skimage.measure import compare_psnr, compare_ssim


# 网络参数量
def print_network_parameters(net: torch.nn.Module):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(str(net))
    print('Total number of parameters: {}'.format(num_params))


def find_last_checkpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = file_[file_.rfind('model_') + 6: file_.rfind('.pth')]
            epochs_exist.append(int(result))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch
