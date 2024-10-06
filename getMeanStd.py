import os
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

# 图像数据集路径
dataset_dirs = [
    'Angle20200610/img-20200610'
]

# 定义图像转换（将图像转换为Tensor）
transform = transforms.Compose([
        transforms.Resize((540, 960)),
        transforms.ToTensor(),  # 转换为张量
    ])

# 初始化存储所有图像数据的Tensor
all_images = []

# 遍历每个数据集文件夹中的所有图像
for dataset_dir in dataset_dirs:
    for root, _, files in os.walk(dataset_dir):
        for file in tqdm(files):
            if file.endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
                # 加载图像并应用转换
                img_path = os.path.join(root, file)
                image = Image.open(img_path).convert('RGB')
                image = transform(image)
                all_images.append(image)

# 将所有图像堆叠为一个大的Tensor
all_images = torch.stack(all_images)

# 计算每个通道的均值和标准差
mean = all_images.mean(dim=[0, 2, 3])
std = all_images.std(dim=[0, 2, 3])

# 输出结果
print(f"Mean: {mean.tolist()}")
print(f"Std: {std.tolist()}")
