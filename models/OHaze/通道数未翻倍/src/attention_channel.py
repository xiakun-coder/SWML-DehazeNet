import torch
import torch.nn as nn



# #Create an Efficent Channel Attention Class
# class Efficient_Channel_Attention(nn.Module):
#
#     def __init__(self, channel, k_size=3): # Initialize the class
#         super(Efficient_Channel_Attention, self).__init__() # inherit from nn.Module
#         self.avg_pool = nn.AdaptiveAvgPool2d(1) # Create an adaptive average pooling layer
#         self.conv=nn.Conv1d(1,1,kernel_size=k_size, padding=((k_size-1)//2), bias=False) # Create a convolutional layer
#         self.sigmoid=nn.Sigmoid() # Create a sigmoid activation layer
#
#     def forward(self, x): # Forward pass of the function
#         y=self.avg_pool(x) # Apply average pooling layer
#         y=y.squeeze(-1).permute(0,2,1) # Change the dimensions of the tensor
#         y=self.conv(y)
#         y=self.sigmoid(y) # Apply sigmoid activation
#         y=y.permute(0,2,1).unsqueeze(-1) # Change the dimensions of the tensor
#
#         return x*y.expand_as(x) # Multiply elementwise

class Efficient_Channel_Attention(nn.Module):
    def __init__(self, channel, k_size=3):
        super().__init__()

        # 全局平均池化 -> (B, C, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # LayerNorm 提升稳定性
        self.ln = nn.LayerNorm(channel)

        # 更强的非线性 channel MLP
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.SiLU(),
            nn.Linear(channel, channel)
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # 1) Global Avg Pool
        y = self.avg_pool(x).view(b, c)

        # 2) LayerNorm 归一化（训练更稳定）
        y_norm = self.ln(y)

        # 3) Squared Attention（增强高响应通道）
        y_sq = y_norm * y_norm

        # 4) 通过 MLP（提升表达能力）
        y = self.fc(y_sq)

        # 5) Sigmoid 输出通道权重
        y = torch.sigmoid(y).view(b, c, 1, 1)

        # 6) 广播乘法 —— 保持你原来的写法
        return x * y.expand_as(x)


