import torch
import torch.nn as nn
import FCB_class
import Attention_Block
import attention_channel
import attention_pixel
import attention_spatial
import torch.nn.functional as F


class FEAB(nn.Module):
    def __init__(self, n_inputs=32, n_outputs=32, k_size=3, bias=False):
        super(FEAB, self).__init__()
        # 保持原有参数，支持输入输出通道数独立设置
        self.fcb1 = FCB_class.FCB(n_inputs, n_outputs, bias)
        self.conv1x1 = nn.Conv2d(n_outputs * 2, n_outputs, kernel_size=1, stride=1, padding=0, bias=bias)
        self.instance_norm = nn.InstanceNorm2d(n_outputs)
        self.Attention_Block = Attention_Block.Attn_Block(n_outputs, k_size)

    def forward(self, x):
        x1 = self.fcb1(x)
        attention_output = self.Attention_Block(x1)
        x5 = self.conv1x1(attention_output)
        x_norm = self.instance_norm(x)
        # 处理可能的通道数不匹配问题：如果输入输出通道数不同，调整norm后的通道数
        if x_norm.shape[1] != x5.shape[1]:
            x_norm = nn.Conv2d(x_norm.shape[1], x5.shape[1], 1, bias=False).to(x_norm.device)(x_norm)
        x_res = x5 + x_norm
        return x_res


class HFP(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(HFP, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        # 改为输出指定的out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0)
        # InstanceNorm使用输出通道数
        self.insnorm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.pad(x)
        x2 = self.conv(x1)
        x3 = self.insnorm(x2)
        x4 = self.relu(x3)
        # 处理残差连接时的通道数不匹配
        if x4.shape[1] != x.shape[1]:
            x = nn.Conv2d(x.shape[1], x4.shape[1], 1, bias=False).to(x.device)(x)
        x4 = x4 + x
        return x4


class HF_Down(nn.Module):
    def __init__(self, channels, out_channels=None, sampling=2):
        super(HF_Down, self).__init__()
        # 新增out_channels参数，默认和输入通道数相同
        self.out_channels = out_channels if out_channels is not None else channels
        self.pad = nn.ReflectionPad2d(1)
        # 卷积层改为输出指定的out_channels
        self.conv = nn.Conv2d(channels, self.out_channels, kernel_size=3, stride=sampling, padding=0)
        self.fcb = FCB_class.FCB(self.out_channels, self.out_channels)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.fcb(x)
        return x


class HF_Up(nn.Module):
    def __init__(self, channels, out_channels=None):
        super(HF_Up, self).__init__()
        # 新增out_channels参数，默认和输入通道数相同
        self.out_channels = out_channels if out_channels is not None else channels
        # FCB层适配输出通道数
        self.fcb_no_act = FCB_class.FCB_No_Act(channels, channels)
        self.pixel_attention = attention_pixel.Efficient_Pixel_Attention(channels)
        # 转置卷积改为输出指定的out_channels
        self.up = nn.ConvTranspose2d(channels, self.out_channels, kernel_size=3, stride=2, padding=1, output_padding=1,
                                     bias=False)
        self.fcb2 = FCB_class.FCB(self.out_channels, self.out_channels)

    def forward(self, x2_h, x2_l):
        x = x2_h + x2_l
        x = self.fcb_no_act(x)
        x = self.pixel_attention(x)
        x = self.up(x)
        x = self.fcb2(x)
        return x


class Fusion_Up(nn.Module):
    def __init__(self, channels, out_channels=None):
        super(Fusion_Up, self).__init__()
        # 新增out_channels参数，默认和输入通道数相同
        self.out_channels = out_channels if out_channels is not None else channels
        # FCB层适配输入通道数
        self.fcb_no_act = FCB_class.FCB_No_Act(channels, channels)
        self.pixel_attention = attention_pixel.Efficient_Pixel_Attention(channels)
        # 转置卷积改为输出指定的out_channels
        self.up = nn.ConvTranspose2d(channels, self.out_channels, kernel_size=3, stride=2, padding=1, output_padding=1,
                                     bias=False)
        self.fcb2 = FCB_class.FCB(self.out_channels, self.out_channels)

    def forward(self, x_prev, x_hf, x_lf):
        x = x_prev + x_hf
        x = x + x_lf
        x = self.fcb_no_act(x)
        x = self.pixel_attention(x)
        x = self.up(x)
        x = self.fcb2(x)
        return x


class FEAG(nn.Module):
    def __init__(self, n_FEAB_Blocks=3, channels=32):
        super(FEAG, self).__init__()
        # FEAB的输入输出都使用指定的channels，适配每层的通道数
        self.FEAB_Blocks = nn.ModuleList([FEAB(n_inputs=channels, n_outputs=channels) for i in range(n_FEAB_Blocks)])

    def forward(self, x):
        for block in self.FEAB_Blocks:
            x = block(x)
        return x


def normalize(tensor):
    return tensor * 2 - 1


def denormalize(tensor):
    return (tensor + 1) / 2