import torch
from torch import nn


class Dual_HDRNet(nn.Module):
    def __init__(self, in_channels=7, basic_channels=64):
        super().__init__()
        self.encoder = Attention_Encoder(in_channels=in_channels, basic_channels=basic_channels)
        self.decoder = Merge_Decoder(basic_channels=basic_channels)

    def forward(self, X1, X2):
        encode_result, Zr = self.encoder(X1, X2)
        out = self.decoder(encode_result, Zr)

        return out


class Attention_Encoder(nn.Module):
    def __init__(self, in_channels=7, basic_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=basic_channels,
                kernel_size=3,
                stride=1, padding=1
            ),
            ResnetBlock(in_channels=basic_channels)
        )

        self.cross_attention = CrossLinearAttention(dim=basic_channels)

    def forward(self, X1, X2):
        Z1 = self.encoder(X1)
        Z2 = self.encoder(X2)
        Atten = self.cross_attention(Z1, Z2)

        encode_result = torch.cat([Z1, Atten, Z2], dim=1)
        return encode_result, Z2


class Merge_Decoder(nn.Module):
    def __init__(self, basic_channels):
        super().__init__()
        self.in_conv = nn.Conv2d(
            in_channels=basic_channels * 3,
            out_channels=basic_channels,
            kernel_size=3,
            stride=1, padding=1
        )
        self.decoder1 = nn.Sequential(
            ResnetBlock(basic_channels),
        )
        self.decoder2 = nn.Sequential(
            ResnetBlock(basic_channels),
        )
        self.decoder3 = nn.Sequential(
            ResnetBlock(basic_channels),
        )

        self.out_conv1 = Block(
            in_channels=basic_channels * 3,
            out_channels=basic_channels
        )
        self.out_conv2 = Block(
            in_channels=basic_channels,
            out_channels=basic_channels
        )
        self.out_conv3 = Block(
            in_channels=basic_channels,
            out_channels=3
        )

    def forward(self, encode_result, Zr):
        F0 = self.in_conv(encode_result)

        F1 = self.decoder1(F0)
        F2 = self.decoder2(F1)
        F3 = self.decoder3(F2)

        F4 = torch.cat([F1, F2, F3], dim=1)

        F5 = self.out_conv1(F4)
        F6 = self.out_conv2(F5 + Zr)
        F7 = self.out_conv3(F6)

        return torch.tanh(F7) * 0.5 + 0.5


# 注意力模块
class CrossLinearAttention(nn.Module):
    """
    https://github.com/lucidrains/linear-attention-transformer
    """
    def __init__(self, dim, heads=4, num_head_channels=32):
        super().__init__()
        # 缩放系数
        self.scale = num_head_channels ** -0.5
        # 头数
        self.heads = heads
        self.num_head_channels = num_head_channels
        # qkv维度
        hidden_dim = num_head_channels * heads

        self.to_q = nn.Conv2d(
            in_channels=dim,
            out_channels=hidden_dim,
            kernel_size=1,
            stride=1, padding=0,
            bias=False
        )
        self.to_k = nn.Conv2d(
            in_channels=dim,
            out_channels=hidden_dim,
            kernel_size=1,
            stride=1, padding=0,
            bias=False
        )
        self.to_v = nn.Conv2d(
            in_channels=dim,
            out_channels=hidden_dim,
            kernel_size=1,
            stride=1, padding=0,
            bias=False
        )

        self.to_out = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=dim,
                kernel_size=1,
                stride=1, padding=0
            ),
            nn.GroupNorm(num_groups=1, num_channels=dim)
        )

    def forward(self, x, y):
        b, c, hight, width = x.shape

        # [b, c, h, w] -> [b, head x head_dim, h, w] -> [b x head, head_dim, h x w]
        q = self.to_q(x).view(b * self.heads, self.num_head_channels, hight * width)
        k = self.to_k(y).view(b * self.heads, self.num_head_channels, hight * width)
        v = self.to_v(y).view(b * self.heads, self.num_head_channels, hight * width)

        # 缩放
        q = q.softmax(dim=-2) * self.scale
        k = k.softmax(dim=-1)

        # [b x head, head_dim_k, h x w] x [b x head, h x w, head_dim_v]
        # = [b x head, head_dim_k, head_dim_v]
        context = torch.bmm(k, v.permute(0, 2, 1))
        # [b x head, head_dim_v, head_dim_k] x [b x head, head_dim, h x w]
        # = [b x head, head_dim_v, h x w]
        output = torch.bmm(context.permute(0, 2, 1), q)
        output = output.view(b, self.heads * self.num_head_channels, hight, width)

        return self.to_out(output) + x


class LinearAttention(nn.Module):
    """
    https://github.com/lucidrains/linear-attention-transformer
    """
    def __init__(self, dim, heads=4, num_head_channels=32):
        super().__init__()
        # 缩放系数
        self.scale = num_head_channels ** -0.5
        # 头数
        self.heads = heads
        self.num_head_channels = num_head_channels
        # qkv维度
        hidden_dim = num_head_channels * heads

        self.to_q = nn.Conv2d(
            in_channels=dim,
            out_channels=hidden_dim,
            kernel_size=1,
            stride=1, padding=0,
            bias=False
        )
        self.to_k = nn.Conv2d(
            in_channels=dim,
            out_channels=hidden_dim,
            kernel_size=1,
            stride=1, padding=0,
            bias=False
        )
        self.to_v = nn.Conv2d(
            in_channels=dim,
            out_channels=hidden_dim,
            kernel_size=1,
            stride=1, padding=0,
            bias=False
        )

        self.to_out = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=dim,
                kernel_size=1,
                stride=1, padding=0
            ),
            nn.GroupNorm(num_groups=1, num_channels=dim)
        )

    def forward(self, x):
        b, c, hight, width = x.shape

        # [b, c, h, w] -> [b, head x head_dim, h, w] -> [b x head, head_dim, h x w]
        q = self.to_q(x).view(b * self.heads, self.num_head_channels, hight * width)
        k = self.to_k(x).view(b * self.heads, self.num_head_channels, hight * width)
        v = self.to_v(x).view(b * self.heads, self.num_head_channels, hight * width)

        # 缩放
        q = q.softmax(dim=-2) * self.scale
        k = k.softmax(dim=-1)

        # [b x head, head_dim_k, h x w] x [b x head, h x w, head_dim_v]
        # = [b x head, head_dim_k, head_dim_v]
        context = torch.bmm(k, v.permute(0, 2, 1))
        # [b x head, head_dim_v, head_dim_k] x [b x head, head_dim, h x w]
        # = [b x head, head_dim_v, h x w]
        output = torch.bmm(context.permute(0, 2, 1), q)
        output = output.view(b, self.heads * self.num_head_channels, hight, width)

        return self.to_out(output) + x


# ResnetBlock的子模块
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, use_act=True):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.act = nn.SiLU() if use_act else nn.Identity()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1, padding=1
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut

        self.block1 = Block(in_channels=self.in_channels, out_channels=self.out_channels)
        self.block2 = Block(in_channels=self.out_channels, out_channels=self.out_channels)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1, padding=1
                )
            else:
                self.nin_shortcut = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1, padding=0
                )

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    device = torch.device("cuda")
    log_name = "Dual_HDRNet_LA"
    sw = SummaryWriter("./logs/model/" + log_name)

    model = Dual_HDRNet().to(device)
    print("Dual_HDRNet have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    image1 = torch.randn(5, 7, 256, 256).to(device)
    image2 = torch.randn(5, 7, 256, 256).to(device)

    out = model(image1, image2)
    print(out.shape)
    print(out.max(), out.min())
    sw.add_graph(model, [image1, image2])
    sw.close()











