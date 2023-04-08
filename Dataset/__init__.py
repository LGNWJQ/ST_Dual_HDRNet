import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

import utils
import numpy as np
import os
import cv2
import imageio
imageio.plugins.freeimage.download()


class HDR_Dataset(Dataset):
    def __init__(self, data_path, patch_size=256, is_Training=True):
        self.data_path = data_path
        data_list = os.listdir(self.data_path)
        # 通过检查是否有hdr文件来过滤无效文件or文件夹
        self.data_list = [x for x in data_list if os.path.exists(os.path.join(self.data_path, x, 'HDRImg.hdr'))]
        self.ToFloat32 = A.ToFloat(max_value=65535.0)
        self.train_transform = A.Compose(
            [
                A.RandomCrop(width=patch_size, height=patch_size),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                ToTensorV2(p=1.0),
            ],
            additional_targets={
                "image1": "image",
                "image2": "image",
            },
        )
        self.test_transform = A.Compose(
            [
                A.CenterCrop(height=512, width=512),
                ToTensorV2(p=1.0),
            ],
            additional_targets={
                "image1": "image",
                "image2": "image",
            },
        )
        self.transform = self.train_transform if is_Training else self.test_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_root = os.path.join(self.data_path, self.data_list[idx] + '/')
        file_list = sorted(os.listdir(data_root))

        # 子文件路径
        LDR_path1 = os.path.join(data_root, file_list[0])
        LDR_path2 = os.path.join(data_root, file_list[1])
        HDR_path = os.path.join(data_root, file_list[2])
        # txt_path = os.path.join(data_root, file_list[3])

        # 读取输入的TIFF图像
        LDR_image1 = cv2.imread(LDR_path1, cv2.IMREAD_UNCHANGED)
        LDR_image1 = cv2.cvtColor(LDR_image1, cv2.COLOR_BGR2RGB)
        LDR_image1 = self.ToFloat32(image=LDR_image1)['image']

        LDR_image2 = cv2.imread(LDR_path2, cv2.IMREAD_UNCHANGED)
        LDR_image2 = cv2.cvtColor(LDR_image2, cv2.COLOR_BGR2RGB)
        LDR_image2 = self.ToFloat32(image=LDR_image2)['image']

        # 读取曝光时间
        # expoTimes = np.power(2, np.loadtxt(txt_path))
        # expoTimes = torch.from_numpy(expoTimes)

        # 读取HDR图像
        HDR_image = np.array(imageio.v2.imread(HDR_path, format='HDR-FI'))
        HDR_image = HDR_image[:, :, (2, 1, 0)]

        # 数据增强
        augmentations = self.transform(
            image=HDR_image,
            image1=LDR_image1,
            image2=LDR_image2,
        )
        LDR_image1 = augmentations['image1']
        LDR_image2 = augmentations['image2']
        HDR_image = augmentations['image']

        batch = {
            'LDR_1': LDR_image1,
            'LDR_2': LDR_image2,
            'HDR': HDR_image
        }

        return batch


from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter
from config import HDR_config
if __name__ == '__main__':
    config = HDR_config()

    HDR_dataset = HDR_Dataset(
        data_path=config.data_path,
        patch_size=512,
        is_Training=False
    )

    data_loader = DataLoader(
        dataset=HDR_dataset,
        batch_size=16,
        num_workers=4,
        shuffle=False
    )

    device = torch.device('cuda')

    ST = utils.Structure_Tensor().to(device)

    for i, sample in enumerate(data_loader):
        print('=-'*30)
        LDR_1 = sample['LDR_1'].to(device)
        LDR_2 = sample['LDR_2'].to(device)
        HDR = sample['HDR'].to(device)

        img_grid_L1 = make_grid(LDR_1, normalize=False, nrow=4)
        img_grid_L2 = make_grid(LDR_2, normalize=False, nrow=4)
        img_grid_H = make_grid(HDR, normalize=False, nrow=4)
        # img_grid_H = utils.Gamma_Correction(img_grid_H)

        # save_image(img_grid_L1, "./L1.png")
        # save_image(img_grid_L2, "./L2.png")
        # save_image(img_grid_H, "./H.png")

        img_grid_H = img_grid_H.detach().cpu().permute(1, 2, 0).numpy()
        print(img_grid_H.shape)
        cv2.imwrite("./HDR.hdr", img_grid_H)

        break

        # # 处理
        # Hu = utils.Mu_Law(H)
        #
        # # 计算结构张量
        # st_1 = ST(X1)
        # st_2 = ST(X2)
        # st_h = ST(H)
        # st_hu = ST(Hu)
        #
        # img_grid_I1 = make_grid(st_1, normalize=False, nrow=3)
        # img_grid_I2 = make_grid(st_2, normalize=False, nrow=3)
        # img_grid_H = make_grid(st_h, normalize=False, nrow=3)
        # img_grid_Hu = make_grid(st_hu, normalize=False, nrow=3)
        #
        # writer.add_image('Input1', img_grid_I1, global_step=i)
        # writer.add_image('Input2', img_grid_I2, global_step=i)
        # writer.add_image('HDR', img_grid_H, global_step=i)
        # writer.add_image('HDR_u_law', img_grid_Hu, global_step=i)






















