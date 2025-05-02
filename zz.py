# from PIL import Image






# image = Image.open('dataset/train/2929_jpg.rf.aef8778dc395eb560b70111d69a77330.jpg')

from PIL import Image

# 1. 打开图片
image_path = 'dataset/train/2929_jpg.rf.aef8778dc395eb560b70111d69a77330.jpg'
image = Image.open(image_path)

# # 2. 处理图片（示例：调整大小、转灰度、旋转等）
# # (1) 调整大小（缩放到 256x256）
# image_resized = image.resize((256, 256))

# # (2) 转换为灰度图（可选）
# image_gray = image.convert('L')

# # (3) 旋转 90 度（可选）
# image_rotated = image.rotate(90)


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
    mask = nn.interpolate(mask.unsqueeze(0), size=(256, 256), mode='nearest').squeeze(0)


# 3. 保存处理后的图片
output_path = 'processed_image.jpg'  # 保存路径
image_resized.save(output_path)  # 保存调整大小后的图片