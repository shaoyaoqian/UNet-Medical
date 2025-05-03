import torch
import torch.nn as nn
import torch.nn.functional as F

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0, pooling=True):
        super(DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        
        # 更平滑的下采样选择
        if pooling:
            # 使用平均池化或带步长的卷积
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1)
            )
        else:
            self.downsample = None
            
        self.dropout = nn.Dropout2d(dropout_prob) if dropout_prob > 0 else None

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        if self.dropout:
            x = self.dropout(x)
        skip = x
        if self.downsample:
            x = self.downsample(x)
        return x, skip

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        # 使用双线性上采样+卷积替代转置卷积，减少棋盘伪影
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.conv(x)  # 调整通道数
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, n_filters=32):
        super(UNet, self).__init__()
        
        # 编码器路径
        self.down1 = DownBlock(n_channels, n_filters)
        self.down2 = DownBlock(n_filters, n_filters * 2)
        self.down3 = DownBlock(n_filters * 2, n_filters * 4, dropout_prob=0.2)
        self.down4 = DownBlock(n_filters * 4, n_filters * 8, dropout_prob=0.2)
        self.down5 = DownBlock(n_filters * 8, n_filters * 16, dropout_prob=0.3)
        
        # 瓶颈层
        self.bottleneck = DownBlock(n_filters * 16, n_filters * 32, dropout_prob=0.4, pooling=False)
        
        # 解码器路径
        self.up1 = UpBlock(n_filters * 32, n_filters * 16)
        self.up2 = UpBlock(n_filters * 16, n_filters * 8)
        self.up3 = UpBlock(n_filters * 8, n_filters * 4)
        self.up4 = UpBlock(n_filters * 4, n_filters * 2)
        self.up5 = UpBlock(n_filters * 2, n_filters)
        
        # 输出层改进
        self.outc = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, 3, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.LeakyReLU(0.1),
            nn.Conv2d(n_filters, n_classes, 1)
        )
        
        # 根据输出范围选择激活函数
        # 方案1: 如果输出在[0,1]范围
        # self.final_activation = nn.Sigmoid()  
        
        # 方案2: 如果输出需要更广范围
        self.final_activation = nn.Identity()  # 无激活，让网络学习任意范围
        
        # 方案3: 确保正输出
        # self.final_activation = nn.Softplus()

    def forward(self, x):
        # 编码器路径
        x1, skip1 = self.down1(x)
        x2, skip2 = self.down2(x1)
        x3, skip3 = self.down3(x2)
        x4, skip4 = self.down4(x3)
        x5, skip5 = self.down5(x4)
        
        # 瓶颈层
        x6, _ = self.bottleneck(x5)
        
        # 解码器路径
        x = self.up1(x6, skip5)
        x = self.up2(x, skip4)
        x = self.up3(x, skip3)
        x = self.up4(x, skip2)
        x = self.up5(x, skip1)
        
        x = self.outc(x)
        x = self.final_activation(x)
        return x