import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import random
import swanlab
from net import UNet
from data import COCOSegmentationDataset


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
        experiment_name="bs32-epoch40",
        config={
            "batch_size": 32,
            "learning_rate": 1e-4,
            "num_epochs": 40,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
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
        indices = indices[:8]
        
        # 创建一个大图
        plt.figure(figsize=(15, 8))  # 调整图像大小以适应新增的行
        plt.suptitle(f'Epoch {swanlab.config["num_epochs"]} Predictions (Random 6 samples)')
        
        for i, idx in enumerate(indices):
            # 原始图像
            plt.subplot(4, 8, i*4 + 1)  # 4行而不是3行
            img = images[idx].cpu().numpy().transpose(1, 2, 0)
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)
            plt.imshow(img)
            plt.title('Original Image')
            plt.axis('off')
            
            # 真实掩码
            plt.subplot(4, 8, i*4 + 2)
            plt.imshow(masks[idx].cpu().squeeze(), cmap='gray')
            plt.title('True Mask')
            plt.axis('off')
            
            # 预测掩码
            plt.subplot(4, 8, i*4 + 3)
            plt.imshow(binary_predictions[idx].cpu().squeeze(), cmap='gray')
            plt.title('Predicted Mask')
            plt.axis('off')

            # 新增：预测掩码叠加在原图上
            plt.subplot(4, 8, i*4 + 4)
            plt.imshow(img)  # 先显示原图
            # 添加红色半透明掩码
            plt.imshow(binary_predictions[idx].cpu().squeeze(), 
                      cmap='Reds', alpha=0.3)  # alpha控制透明度
            plt.title('Overlay')
            plt.axis('off')
        
        # 记录图像到SwanLab
        swanlab.log({"predictions": swanlab.Image(plt)})

if __name__ == '__main__':
    main()
