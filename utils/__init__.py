import os

import torch
from torch import nn
import numpy as np
import random


def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def print_info(str_info, end=False):
    print("=-" * 50)
    print(str_info)
    if end:
        print("=-" * 50)


def Mu_Law(image_tensor, mu=5000):
    device = image_tensor.device
    mu = torch.tensor([mu]).to(device)
    return torch.log(1 + image_tensor*mu) / torch.log(1 + mu)


def Gamma_Correction(image_tensor, gamma=2.2):
    return torch.pow(image_tensor, 1.0/gamma)


def PSNR(image, label):
    mse = nn.functional.mse_loss(image, label)
    return 10 * torch.log10(1 / mse)


def del_file(path, threshold=5):
    file_list = os.listdir(path)
    num_file = len(file_list)
    if num_file > threshold:
        del_file_list = file_list[:-threshold]
        for file in del_file_list:
            file_path = os.path.join(path, file)
            os.remove(file_path)


class Structure_Tensor(nn.Module):
    def __init__(self):
        super(Structure_Tensor, self).__init__()
        self.gradient_X = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1),
            padding_mode='reflect'
        )
        self.X_kernel = torch.tensor([-0.5, 0, 0.5], dtype=torch.float32).view(1, 1, 1, 3)
        self.gradient_X.weight.data = self.X_kernel

        self.gradient_Y = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(3, 1),
            stride=(1, 1),
            padding=(1, 0),
            padding_mode='reflect'
        )
        self.Y_kernel = torch.tensor([-0.5, 0, 0.5], dtype=torch.float32).view(1, 1, 3, 1)
        self.gradient_Y.weight.data = self.Y_kernel

    def forward(self, x):
        # 计算灰度图
        r, g, b = x.unbind(dim=-3)
        gray = (0.2989 * r + 0.587 * g + 0.114 * b)
        gray = gray.unsqueeze(dim=-3) * 255.0

        # 计算梯度
        Ix = self.gradient_X(gray)
        Iy = self.gradient_Y(gray)

        Ix2 = torch.pow(Ix, 2)
        Iy2 = torch.pow(Iy, 2)
        Ixy = Ix * Iy

        # 计算行列式和迹
        #  Ix2, Ixy
        #  Ixy, Iy2
        H = Ix2 + Iy2
        K = Ix2 * Iy2 - Ixy * Ixy

        # Flat平坦区域：H = 0;
        # Edge边缘区域：H > 0 & & K = 0;
        # Corner角点区域：H > 0 & & K > 0;

        h_ = 100

        Flat = torch.zeros_like(H)
        Flat[H < h_] = 1.0

        Edge = torch.zeros_like(H)
        Edge[(H >= h_) * (K.abs() <= 1e-6)] = 1.0

        Corner = torch.zeros_like(H)
        Corner[(H >= h_) * (K.abs() > 1e-6)] = 1.0

        return 1.0 - Flat, torch.cat([Ix2, Ixy, Iy2, Ixy], dim=1)


if __name__ == "__main__":
    # path = "D:/IEEE_SPL/Experiment/HDR_0.1/checkpoint/"
    #
    # file_list = os.listdir(path)
    # del_file_list = file_list[:-5]
    #
    # for file in del_file_list:
    #     file_path = os.path.join(path, file)
    #     os.remove(file_path)
    #
    # num = len(file_list)
    #
    # print(num)
    device = torch.device("cuda")
    ST = Structure_Tensor().to(device)
    img = torch.randn(5, 3, 256, 256).to(device)

    out1, out2 = ST(img)
    print(out1.shape)
    print(out2.shape)
















