import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import swanlab

# 数据路径设置
train_dir = './dataset/train'
val_dir = './dataset/valid'
test_dir = './dataset/test'

train_annotation_file = './dataset/train/_annotations.coco.json'
test_annotation_file = './dataset/test/_annotations.coco.json'
val_annotation_file = './dataset/valid/_annotations.coco.json'

# 加载COCO数据集
train_coco = COCO(train_annotation_file)
val_coco = COCO(val_annotation_file)
test_coco = COCO(test_annotation_file)

class COCOSegmentationDataset(Dataset):
    def __init__(self, coco, image_dir, transform=None):
        self.coco = coco
        self.image_dir = image_dir
        self.image_ids = coco.getImgIds()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])

        # 加载图像
        image = Image.open(image_path)
        image = np.array(image, dtype=np.uint8)

        # 创建掩码
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
        for ann in anns:
            mask = np.maximum(mask, self.coco.annToMask(ann))

        # 转换为张量并预处理
        if self.transform:
            image = self.transform(image)
            mask = torch.from_numpy(mask).float().unsqueeze(0)
            mask = nn.functional.interpolate(mask.unsqueeze(0), size=(256, 256), mode='nearest').squeeze(0)

        return image, mask

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
    def __init__(self, n_channels=3, n_classes=1, n_filters=32):
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

# 定义损失函数
def dice_loss(pred, target, smooth=1e-6):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - ((2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))

def combined_loss(pred, target):
    dice = dice_loss(pred, target)
    bce = nn.BCELoss()(pred, target)
    return 0.6 * dice + 0.4 * bce

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_loss = float('inf')
    patience = 8
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += (outputs.round() == masks).float().mean().item()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        val_acc = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_acc += (outputs.round() == masks).float().mean().item()
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        swanlab.log(
            {
                "train/loss": train_loss,
                "train/acc": train_acc,
                "train/epoch": epoch+1,
                "val/loss": val_loss,
                "val/acc": val_acc,
            },
            step=epoch+1)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

def main():
    swanlab.init(
        project="Unet-Medical-Segmentation",
        experiment_name="Unet",
        config={
            "batch_size": 32,
            "learning_rate": 1e-4,
            "num_epochs": 40,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
    )
    
    # 设置设备
    device = torch.device(swanlab.config["device"])
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = COCOSegmentationDataset(train_coco, train_dir, transform=transform)
    val_dataset = COCOSegmentationDataset(val_coco, val_dir, transform=transform)
    test_dataset = COCOSegmentationDataset(test_coco, test_dir, transform=transform)
    
    # 创建数据加载器
    BATCH_SIZE = swanlab.config["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # 初始化模型
    model = UNet(n_filters=32).to(device)
    
    # 设置优化器和学习率
    optimizer = optim.Adam(model.parameters(), lr=swanlab.config["learning_rate"])
    
    # 训练模型
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=combined_loss,
        optimizer=optimizer,
        num_epochs=swanlab.config["num_epochs"],
        device=device,
    )
    
    # 在测试集上评估
    model.eval()
    test_loss = 0
    test_acc = 0
    
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = combined_loss(outputs, masks)
            test_loss += loss.item()
            test_acc += (outputs.round() == masks).float().mean().item()
    
    test_loss /= len(test_loader)
    test_acc /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    swanlab.log({"test/loss": test_loss, "test/acc": test_acc})
    
    # 可视化预测结果
    visualize_predictions(model, test_loader, device, num_samples=10)
    

def visualize_predictions(model, test_loader, device, num_samples=5, threshold=0.5):
    model.eval()
    with torch.no_grad():
        # 获取一个批次的数据
        images, masks = next(iter(test_loader))
        images, masks = images.to(device), masks.to(device)
        predictions = model(images)
        
        # 将预测结果转换为二值掩码
        binary_predictions = (predictions > threshold).float()
        
        # 选择前3个样本
        indices = random.sample(range(len(images)), min(num_samples, len(images)))
        indices = indices[:6]
        
        # 创建一个大图
        plt.figure(figsize=(15, 10))
        plt.suptitle(f'Epoch {swanlab.config["num_epochs"]+1} Predictions (Random 6 samples)')
        
        for i, idx in enumerate(indices):
            # 原始图像
            plt.subplot(3, 6, i*3 + 1)
            img = images[idx].cpu().numpy().transpose(1, 2, 0)
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)
            plt.imshow(img)
            plt.title('Original Image')
            plt.axis('off')
            
            # 真实掩码
            plt.subplot(3, 6, i*3 + 2)
            plt.imshow(masks[idx].cpu().squeeze(), cmap='gray')
            plt.title('True Mask')
            plt.axis('off')
            
            # 预测掩码
            plt.subplot(3, 6, i*3 + 3)
            plt.imshow(binary_predictions[idx].cpu().squeeze(), cmap='gray')
            plt.title('Predicted Mask')
            plt.axis('off')
        
        # plt.tight_layout()
        # 保存到本地
        plt.savefig('./predictions.png')
        swanlab.log({"predictions": swanlab.Image(plt)})

if __name__ == '__main__':
    main()
