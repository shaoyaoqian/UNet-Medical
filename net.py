import torch
import torch.nn as nn

# 定义U-Net模型的下采样块
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0, max_pooling=True):
        super(DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2) if max_pooling else None
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else None

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        if self.dropout:
            x = self.dropout(x)
        skip = x
        if self.maxpool:
            x = self.maxpool(x)
        return x, skip

# 定义U-Net模型的上采样块
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

# 定义完整的U-Net模型
class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, n_filters=32):
        super(UNet, self).__init__()
        
        # 编码器路径
        self.down1 = DownBlock(n_channels, n_filters)
        self.down2 = DownBlock(n_filters, n_filters * 2)
        self.down3 = DownBlock(n_filters * 2, n_filters * 4)
        self.down4 = DownBlock(n_filters * 4, n_filters * 8)
        self.down5 = DownBlock(n_filters * 8, n_filters * 16)
        
        # 瓶颈层 - 移除最后的maxpooling
        self.bottleneck = DownBlock(n_filters * 16, n_filters * 32, dropout_prob=0.4, max_pooling=False)
        
        # 解码器路径
        self.up1 = UpBlock(n_filters * 32, n_filters * 16)
        self.up2 = UpBlock(n_filters * 16, n_filters * 8)
        self.up3 = UpBlock(n_filters * 8, n_filters * 4)
        self.up4 = UpBlock(n_filters * 4, n_filters * 2)
        self.up5 = UpBlock(n_filters * 2, n_filters)
        
        # 输出层
        self.outc = nn.Conv2d(n_filters, n_classes, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        # 编码器路径
        x1, skip1 = self.down1(x)      # 128
        x2, skip2 = self.down2(x1)     # 64
        x3, skip3 = self.down3(x2)     # 32
        x4, skip4 = self.down4(x3)     # 16
        x5, skip5 = self.down5(x4)     # 8
        
        # 瓶颈层
        x6, skip6 = self.bottleneck(x5)  # 8 (无下采样)
        
        # 解码器路径
        x = self.up1(x6, skip5)    # 16
        x = self.up2(x, skip4)     # 32
        x = self.up3(x, skip3)     # 64
        x = self.up4(x, skip2)     # 128
        x = self.up5(x, skip1)     # 256
        
        x = self.outc(x)
        x = self.sigmoid(x)
        return x