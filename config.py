import argparse


def HDR_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code_writer', type=str, default="LGN-WJQ", help="Name of code writer")

    # 数据相关参数
    # 'F:/dataset_vae/LOL/our485/'
    # 'F:/dataset_vae/total4/'
    parser.add_argument('--train_data_path', type=str, default='D:/IEEE_SPL/HDR_Data/', help='数据集路径')
    parser.add_argument('--val_data_path', type=str, default='D:/IEEE_SPL/HDR_Data/', help='数据集路径')
    parser.add_argument('--patch_size', type=int, default=128, help='训练时裁剪的图像尺寸')
    parser.add_argument('--batch_size', type=int, default=2, help='批量大小')
    parser.add_argument('--num_worker', type=int, default=2, help='读取数据集的cpu线程数量')

    # 训练相关参数
    parser.add_argument('--experiment_name', type=str, default="ST_HDR_0.2", help='本次实验名称')
    parser.add_argument('--ST_parm', type=float, default=1e-4, help='结构张量损失的系数')
    parser.add_argument('--seed', type=int, default=3407, help='随机种子，用于保证实验可复现')
    parser.add_argument('--save_frequence', type=int, default=200, help='保存权重的频率')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='学习率')
    parser.add_argument('--epochs', type=int, default=10000, help='训练周期数')
    parser.add_argument('--resume_path', type=str,
                        default=None,
                        help='继续训练的权重的路径')

    # 网络配置
    parser.add_argument('--basic_channels', type=int, default=128, help='网络通道数量')
    parser.add_argument('--use_struct_tensor', type=bool, default=True, help="是否使用结构张量")
    # 模板：parser.add_argument('--', type=, default=, help='')

    # 显示参数
    args = parser.parse_args()
    print('=-' * 50)
    for arg in vars(args):
        print('--', arg, ':', getattr(args, arg))
    print('=-' * 50)

    return args


import os
if __name__ == '__main__':
    data_path = "HDR_Data/001/"
    file_list = sorted(os.listdir(data_path))
    print(file_list)
