import torch
import torch.nn as nn
import attention_pixel
import DWT_Block
import torch.nn.functional as F
import attention_channel
import Feature_Processing


# 新增模块：低频增强块（4层卷积，可加通道注意力）
class LowFreq_Enhance(nn.Module):
    def __init__(self, ch, use_ca=True):
        super(LowFreq_Enhance, self).__init__()
        layers = []
        for _ in range(4):
            layers += [
                nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True)
            ]
        self.enhance = nn.Sequential(*layers)
        self.use_ca = use_ca
        if use_ca:
            self.ca = attention_channel.Efficient_Channel_Attention(ch)

    def forward(self, x):
        out = self.enhance(x)
        if self.use_ca:
            out = self.ca(out)
        return out


class Freq_Proc_Module(nn.Module):
    def __init__(self, n_FEAB_Blocks=[3,3,3,3], sampling=2, in_ch=3, base_ch=32):
        super(Freq_Proc_Module, self).__init__()

        ch1 = base_ch
        ch2 = base_ch * 2
        ch3 = base_ch * 4
        ch4 = base_ch * 8

        # -------- stage1 --------
        self.split1 = DWT_Block.DWT_Block(in_channels=in_ch, out_channels=ch1, sampling=sampling)
        self.lf_proc_1 = Feature_Processing.FEAG(n_FEAB_Blocks=n_FEAB_Blocks[0], channels=ch1)
        self.hf_proc_1 = Feature_Processing.HFP(in_channels=ch1, out_channels=ch1)
        self.hf_down_1 = Feature_Processing.HF_Down(ch1)

        # -------- stage2 --------
        self.split2 = DWT_Block.DWT_Block(in_channels=ch1, out_channels=ch2, sampling=sampling)
        self.lf_proc_2 = Feature_Processing.FEAG(n_FEAB_Blocks=n_FEAB_Blocks[1], channels=ch2)
        self.hf_proc_2 = Feature_Processing.HFP(in_channels=ch2, out_channels=ch2)
        self.hf_down_2 = Feature_Processing.HF_Down(ch2)

        # -------- stage3 --------
        self.split3 = DWT_Block.DWT_Block(in_channels=ch2, out_channels=ch3, sampling=sampling)
        self.lf_proc_3 = Feature_Processing.FEAG(n_FEAB_Blocks=n_FEAB_Blocks[2], channels=ch3)
        self.hf_proc_3 = Feature_Processing.HFP(in_channels=ch3, out_channels=ch3)
        self.hf_down_3 = Feature_Processing.HF_Down(ch3)

        # -------- stage4 --------
        self.split4 = DWT_Block.DWT_Block(in_channels=ch3, out_channels=ch4, sampling=sampling)
        self.lf_proc_4 = Feature_Processing.FEAG(n_FEAB_Blocks=n_FEAB_Blocks[3], channels=ch4)
        self.hf_proc_4 = Feature_Processing.HFP(in_channels=ch4, out_channels=ch4)

        # 通道对齐 1x1 conv
        self.align_1_2 = nn.Conv2d(ch1, ch2, kernel_size=1)
        self.align_2_3 = nn.Conv2d(ch2, ch3, kernel_size=1)
        self.align_3_4 = nn.Conv2d(ch3, ch4, kernel_size=1)

        self.hf_up = Feature_Processing.HF_Up(ch4)

        self.fusion1 = Feature_Processing.Fusion_Up(ch3)
        self.fusion2 = Feature_Processing.Fusion_Up(ch2)
        self.fusion3 = Feature_Processing.Fusion_Up(ch1)

    def forward(self, x):
        # -------- stage1 --------
        x1_l, x1_h = self.split1(x)
        x1_low = self.lf_proc_1(x1_l)
        x1_high = self.hf_proc_1(x1_h)
        x1_high_down = self.hf_down_1(x1_high)

        # -------- stage2 --------
        x2_l, x2_h = self.split2(x1_low)
        x2_low = self.lf_proc_2(x2_l)

        x1_high_down = self.align_1_2(x1_high_down)
        x2_high = self.hf_proc_2(x2_h + x1_high_down)
        x2_high_down = self.hf_down_2(x2_high)

        # -------- stage3 --------
        x3_l, x3_h = self.split3(x2_low)
        x3_low = self.lf_proc_3(x3_l)

        x2_high_down = self.align_2_3(x2_high_down)
        x3_high = self.hf_proc_3(x3_h + x2_high_down)
        x3_high_down = self.hf_down_3(x3_high)

        # -------- stage4 --------
        x4_l, x4_h = self.split4(x3_low)
        x4_low = self.lf_proc_4(x4_l)

        x3_high_down = self.align_3_4(x3_high_down)
        x4_high = self.hf_proc_4(x4_h + x3_high_down)

        # -------- up & fusion --------
        y4 = self.hf_up(x4_high, x4_low)
        y3 = self.fusion1(y4, x3_high, x3_low)
        y2 = self.fusion2(y3, x2_high, x2_low)
        y1 = self.fusion3(y2, x1_high, x1_low)

        return y4, y3, y2, y1





class Post_Proc_Module(nn.Module):
    def __init__(self, in_ch=32):
        super(Post_Proc_Module, self).__init__()
        self.channel_attention = attention_channel.Efficient_Channel_Attention(in_ch)
        self.pixel_attention = attention_pixel.Efficient_Pixel_Attention(in_ch)
        self.final_conv = nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, padding=1)
        self.final_conv1 = nn.Conv2d(in_ch // 2, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x, y):
        x = self.channel_attention(x)
        x = self.pixel_attention(x)
        x = self.final_conv(x)
        x = self.relu(x)
        x = self.final_conv1(x)
        x = x + y
        x = self.tanh(x)
        return x

class ToRGB(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 3, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)

class Dehazing_Model(nn.Module):
    def __init__(self, n_FEAB_Blocks=[3,3,3,3], sampling=2, base_ch=32):
        super(Dehazing_Model, self).__init__()

        self.freq_proc = Freq_Proc_Module(
            n_FEAB_Blocks=n_FEAB_Blocks,
            sampling=sampling,
            in_ch=3,
            base_ch=base_ch
        )

        self.post_proc = Post_Proc_Module(in_ch=base_ch)

        self.head_y4 = ToRGB(base_ch * 8)
        self.head_y3 = ToRGB(base_ch * 4)
        self.head_y2 = ToRGB(base_ch * 2)

    def forward(self, x):
        y4, y3, y2, y1 = self.freq_proc(x)

        y1_rgb = self.post_proc(y1, x)
        y2_rgb = self.head_y2(y2)
        y3_rgb = self.head_y3(y3)
        y4_rgb = self.head_y4(y4)

        return y4_rgb, y3_rgb, y2_rgb, y1_rgb


