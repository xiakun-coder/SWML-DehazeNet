import torch
import torch.nn as nn

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return x_LL, torch.cat((x_HL, x_LH, x_HH), 1)

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False
    def forward(self, x):
        return dwt_init(x)

class DWT_transform(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.dwt = DWT()
        self.conv1x1_low = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv1x1_high = nn.Conv2d(in_channels*3, out_channels, kernel_size=1, padding=0)
    def forward(self, x):
        dwt_low_frequency,dwt_high_frequency = self.dwt(x)
        dwt_low_frequency = self.conv1x1_low(dwt_low_frequency)
        dwt_high_frequency = self.conv1x1_high(dwt_high_frequency)
        return dwt_low_frequency,dwt_high_frequency

class DWT_Block(nn.Module):
    def __init__(self, in_channels, out_channels,sampling):
        super().__init__()
        self.dwt = DWT_transform(in_channels, out_channels)
        self.pad_spatial = nn.ReflectionPad2d(1)
        self.spatial_conv = nn.Conv2d(in_channels, out_channels,stride=sampling, kernel_size=3, padding=0)
        
    
    def forward(self, x):
        x1=self.pad_spatial(x)
        x1=self.spatial_conv(x1)
        x2_low, x2_high = self.dwt(x)
        x2_low = x2_low + x1
        return x2_low, x2_high

class InverseWavelet(nn.Module):
    def __init__(self, channels):
        super(InverseWavelet, self).__init__()
        self.requires_grad = False

        # 将压缩后的高频 C → 3C
        self.expand_high = nn.Conv2d(
            channels, channels * 3, kernel_size=1, bias=False
        )

    def forward(self, x_ll, x_h):
        """
        x_ll: (B, C, H, W)
        x_h : (B, C, H, W)  ← 已被 conv1x1_high 压缩
        """

        x_h = self.expand_high(x_h)   # (B, 3C, H, W)
        x_hl, x_lh, x_hh = torch.chunk(x_h, 3, dim=1)

        x1 = (x_ll - x_hl - x_lh + x_hh) / 2
        x2 = (x_ll - x_hl + x_lh - x_hh) / 2
        x3 = (x_ll + x_hl - x_lh - x_hh) / 2
        x4 = (x_ll + x_hl + x_lh + x_hh) / 2

        B, C, H, W = x_ll.shape
        out = torch.zeros(
            (B, C, H * 2, W * 2),
            device=x_ll.device,
            dtype=x_ll.dtype
        )

        out[:, :, 0::2, 0::2] = x1
        out[:, :, 1::2, 0::2] = x2
        out[:, :, 0::2, 1::2] = x3
        out[:, :, 1::2, 1::2] = x4

        return out