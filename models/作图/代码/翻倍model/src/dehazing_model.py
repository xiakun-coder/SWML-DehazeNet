import torch
import torch.nn as nn
import attention_pixel
import DWT_Block
import torch.nn.functional as F
import attention_channel
import Feature_Processing


class Freq_Proc_Module(nn.Module):
    def __init__(self, n_FEAB_Blocks=[3, 3, 3, 3], sampling=2, in_ch=3, base_ch=32):
        super(Freq_Proc_Module, self).__init__()

        # 通道数按UNet风格翻倍: 32 → 64 → 128 → 256
        ch1 = base_ch  # 第一层通道数
        ch2 = base_ch * 2  # 第二层通道数
        ch3 = base_ch * 4  # 第三层通道数
        ch4 = base_ch * 8  # 第四层通道数

        # 第一层处理：输入3通道 → ch1通道
        self.split1 = DWT_Block.DWT_Block(in_channels=in_ch, out_channels=ch1, sampling=sampling)
        self.lf_proc_1 = Feature_Processing.FEAG(n_FEAB_Blocks=n_FEAB_Blocks[0], channels=ch1)
        self.hf_proc_1 = Feature_Processing.HFP(in_channels=ch1, out_channels=ch1)
        self.hf_down_1 = Feature_Processing.HF_Down(ch1, out_channels=ch2)  # 下采样到下一层通道数

        # 第二层处理：ch1 → ch2
        self.split2 = DWT_Block.DWT_Block(in_channels=ch1, out_channels=ch2, sampling=sampling)
        self.lf_proc_2 = Feature_Processing.FEAG(n_FEAB_Blocks=n_FEAB_Blocks[1], channels=ch2)
        self.hf_proc_2 = Feature_Processing.HFP(in_channels=ch2, out_channels=ch2)
        self.hf_down_2 = Feature_Processing.HF_Down(ch2, out_channels=ch3)  # 下采样到下一层通道数

        # 第三层处理：ch2 → ch3
        self.split3 = DWT_Block.DWT_Block(in_channels=ch2, out_channels=ch3, sampling=sampling)
        self.lf_proc_3 = Feature_Processing.FEAG(n_FEAB_Blocks=n_FEAB_Blocks[2], channels=ch3)
        self.hf_proc_3 = Feature_Processing.HFP(in_channels=ch3, out_channels=ch3)
        self.hf_down_3 = Feature_Processing.HF_Down(ch3, out_channels=ch4)  # 下采样到下一层通道数

        # 第四层处理：ch3 → ch4
        self.split4 = DWT_Block.DWT_Block(in_channels=ch3, out_channels=ch4, sampling=sampling)
        self.lf_proc_4 = Feature_Processing.FEAG(n_FEAB_Blocks=n_FEAB_Blocks[3], channels=ch4)
        self.hf_proc_4 = Feature_Processing.HFP(in_channels=ch4, out_channels=ch4)

        # 上采样和融合模块：通道数从高到低匹配
        self.hf_up = Feature_Processing.HF_Up(ch4, out_channels=ch3)  # 4→3
        self.fusion1 = Feature_Processing.Fusion_Up(ch3, out_channels=ch2)  # 3→2
        self.fusion2 = Feature_Processing.Fusion_Up(ch2, out_channels=ch1)  # 2→1
        self.fusion3 = Feature_Processing.Fusion_Up(ch1, out_channels=base_ch)  # 1→基础通道数

    def forward(self, x):
        # 第一层：3通道 → ch1通道
        x1_l, x1_h = self.split1(x)
        x1_low = self.lf_proc_1(x1_l)
        x1_high = self.hf_proc_1(x1_h)
        x1_high_down = self.hf_down_1(x1_high)  # ch1 → ch2

        # 第二层：ch1 → ch2通道
        x2_l, x2_h = self.split2(x1_low)
        x2_low = self.lf_proc_2(x2_l)
        x2_high_fuse = x2_h + x1_high_down  # 通道数都是ch2，可直接相加
        x2_high = self.hf_proc_2(x2_high_fuse)
        x2_high_down = self.hf_down_2(x2_high)  # ch2 → ch3

        # 第三层：ch2 → ch3通道
        x3_l, x3_h = self.split3(x2_low)
        x3_low = self.lf_proc_3(x3_l)
        x3_high_fuse = x3_h + x2_high_down  # 通道数都是ch3，可直接相加
        x3_high = self.hf_proc_3(x3_high_fuse)
        x3_high_down = self.hf_down_3(x3_high)  # ch3 → ch4

        # 第四层：ch3 → ch4通道
        x4_l, x4_h = self.split4(x3_low)
        x4_low = self.lf_proc_4(x4_l)
        x4_high_fuse = x4_h + x3_high_down  # 通道数都是ch4，可直接相加
        x4_high = self.hf_proc_4(x4_high_fuse)

        # 上采样融合：从高通道数往低通道数回退
        y4 = self.hf_up(x4_high, x4_low)  # ch4 → ch3
        y3 = self.fusion1(y4, x3_high, x3_low)  # ch3 → ch2
        y2 = self.fusion2(y3, x2_high, x2_low)  # ch2 → ch1
        y1 = self.fusion3(y2, x1_high, x1_low)  # ch1 → base_ch

        return y1


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


class Dehazing_Model(nn.Module):
    def __init__(self, n_FEAB_Blocks=[3, 3, 3, 3], sampling=2, base_channels=32):
        super(Dehazing_Model, self).__init__()
        # 传入基础通道数，内部自动翻倍
        self.freq_proc = Freq_Proc_Module(
            n_FEAB_Blocks=n_FEAB_Blocks,
            sampling=sampling,
            in_ch=3,
            base_ch=base_channels
        )
        self.post_proc = Post_Proc_Module(in_ch=base_channels)  # 最终回到基础通道数

    def forward(self, x):
        x1 = self.freq_proc(x)
        x2 = self.post_proc(x1, x)
        return x2


# 测试代码：验证模型是否能正常初始化和前向传播
if __name__ == "__main__":
    # 创建模型实例，基础通道数32，会自动翻倍为32→64→128→256
    model = Dehazing_Model(base_channels=32)
    # 生成测试输入 (batch_size=1, channels=3, height=256, width=256)
    test_input = torch.randn(1, 3, 256, 256)
    # 前向传播
    output = model(test_input)
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")  # 应输出 torch.Size([1, 3, 256, 256])