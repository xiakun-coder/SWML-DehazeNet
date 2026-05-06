import torch
import torch.nn as nn

class Efficient_Channel_Attention(nn.Module):
    def __init__(self, channel, k_size=3):
        super().__init__()

        # 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # LayerNorm
        self.ln = nn.LayerNorm(channel)

        # channel MLP
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.SiLU(),
            nn.Linear(channel, channel)
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # 1) Global Avg Pool
        y = self.avg_pool(x).view(b, c)

        # 2) LayerNorm 归一化
        y_norm = self.ln(y)

        # 3) Squared Attention
        y_sq = y_norm * y_norm

        # 4) 通过 MLP
        y = self.fc(y_sq)

        # 5) Sigmoid
        y = torch.sigmoid(y).view(b, c, 1, 1)

        # 6) 广播乘法
        return x * y.expand_as(x)


