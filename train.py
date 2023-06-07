import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from tqdm import tqdm

import torch
from torch import nn
import torchvision.utils
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler

from config import HDR_config
import utils
from model import Dual_HDRNet
from Dataset import HDR_Dataset


def train(config):
    # 设定随机数种子
    utils.seed_everything(seed=config.seed)

    # 构建文件结构
    root = f"./Experiment/{config.experiment_name}/"
    log_path = os.path.join(root, "log/")
    checkpoint_path = os.path.join(root, "checkpoint/")

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    writer = SummaryWriter(log_path)
    utils.print_info("--Experimental data path creation complete: " + root)

    # 设定计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils.print_info("--Training on：{}".format(torch.cuda.get_device_name(0)))

    # 构建神经网络
    model = Dual_HDRNet().to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=3e-6)
    ST = utils.Structure_Tensor().to(device)

    mse_fun = nn.MSELoss()

    # 载入权重
    pre_epoch = 0
    if config.resume_path is not None:
        checkpoint = torch.load(config.resume_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        pre_epoch = checkpoint['epoch']
        utils.print_info(f"Load Successfully from {config.resume_path}")

    # 导入训练数据集
    train_set = HDR_Dataset(
        data_path=config.train_data_path,
        patch_size=config.patch_size,
        is_Training=True
    )
    val_set = HDR_Dataset(
        data_path=config.val_data_path,
        patch_size=config.patch_size,
        is_Training=False
    )
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=config.batch_size,
        num_workers=config.num_worker,
        shuffle=True
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=4,
        num_workers=config.num_worker,
        shuffle=False
    )

    for i, batch in enumerate(val_loader):
        val_LDR_1 = batch['LDR_1'].to(device)
        val_LDR_2 = batch['LDR_2'].to(device)
        val_HDR = batch['HDR'].to(device)

        val_L1_grid = torchvision.utils.make_grid(
            val_LDR_1, normalize=False, nrow=2
        )
        val_L2_grid = torchvision.utils.make_grid(
            val_LDR_2, normalize=False, nrow=2
        )
        val_HDR_grid = torchvision.utils.make_grid(
            val_HDR, normalize=False, nrow=2
        )

        writer.add_image('Val_LDR_1', val_L1_grid, global_step=i)
        writer.add_image('Val_LDR_2', val_L2_grid, global_step=i)
        writer.add_image('Val_HDR', val_HDR_grid, global_step=i)
        break
    utils.print_info(f"--The val dataset is loaded and the number of images is {val_set.__len__()}")

    now_PSNR = 0.0
    utils.print_info("--Start training...")
    step = 0
    for epoch in range(config.epochs):
        loop = tqdm(train_loader)
        for i, batch in enumerate(loop):
            """
            验证
            """
            if epoch % 2 == 0:
                val_gamma_1 = utils.Gamma_Correction(val_LDR_1)
                val_st_1, _ = ST(val_LDR_1)
                val_X1 = torch.cat([val_LDR_1, val_gamma_1, val_st_1], dim=1)
                val_gamma_2 = utils.Gamma_Correction(val_LDR_2)
                val_st_2, _ = ST(val_LDR_2)
                val_X2 = torch.cat([val_LDR_2, val_gamma_2, val_st_2], dim=1)

                with torch.no_grad():
                    model.eval()
                    val_reconstruction = model(val_X1, val_X2)
                    val_loss = mse_fun(utils.Mu_Law(val_reconstruction), utils.Mu_Law(val_HDR))
                    val_PSNR = utils.PSNR(utils.Mu_Law(val_reconstruction), utils.Mu_Law(val_HDR))
                    model.train()

                writer.add_scalar('Val_MSE_Loss', val_loss.item(), global_step=step)
                writer.add_scalar('Val_PSNR', val_PSNR, global_step=step)
                val_reconstruction_grid = torchvision.utils.make_grid(val_reconstruction, normalize=False, nrow=2)
                writer.add_image('Res HDR', val_reconstruction_grid, global_step=step)

                if val_PSNR > now_PSNR:
                    now_PSNR = val_PSNR
                    save_path = os.path.join(checkpoint_path, f'epoch_{epoch}_psnr_{now_PSNR}.pth')
                    torch.save(
                        {
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': pre_epoch + epoch
                        }, save_path
                    )
                    utils.del_file(checkpoint_path)
            """
            训练
            """
            LDR_1 = batch['LDR_1'].to(device)
            LDR_2 = batch['LDR_2'].to(device)
            HDR = batch['HDR'].to(device)

            gamma_1 = utils.Gamma_Correction(LDR_1)
            st_1, _ = ST(LDR_1)
            X1 = torch.cat([LDR_1, gamma_1, st_1], dim=1)
            gamma_2 = utils.Gamma_Correction(LDR_2)
            st_2, _ = ST(LDR_2)
            X2 = torch.cat([LDR_2, gamma_2, st_2], dim=1)

            reconstruction = model(X1, X2)

            optimizer.zero_grad()
            loss1 = mse_fun(utils.Mu_Law(reconstruction), utils.Mu_Law(HDR))
            _, re_st_tensor = ST(reconstruction)
            _, HDR_st_tensor = ST(HDR)
            loss2 = mse_fun(re_st_tensor, HDR_st_tensor)
            loss2 = torch.sqrt(loss2)

            loss = loss1 + config.ST_parm * loss2

            loss.backward()
            optimizer.step()

            loop.set_description(f"Train Epoch [{epoch}/{config.epochs}]")
            loop.set_postfix(ST_Loss=loss2.item(), MSE_Loss=loss1.item(), All_Loss=loss.item())
            step += 1

        scheduler.step()
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step=epoch)
        writer.add_scalar("Train_Loss", loss.item(), global_step=epoch)
        writer.add_scalar("MSE_Loss", loss1.item(), global_step=epoch)
        writer.add_scalar("ST_Loss", loss2.item(), global_step=epoch)
        if epoch % config.save_frequence == 0:
            save_path = os.path.join(checkpoint_path, f'epoch_{epoch}.pth')
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': pre_epoch + epoch
                }, save_path
            )
        utils.del_file(checkpoint_path)


if __name__ == "__main__":
    config = HDR_config()
    train(config)
