import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# 定义标签的映射
LABEL_MAPPING = {-30: 0, -25: 1, -20: 2, -15: 3, -10: 4, -5: 5, 0: 6, 5: 7}

class SteelDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        self.img_dir = img_dir
        self.label_file = label_file
        self.transform = transform
        self.img_labels = self._load_labels()

    def _load_labels(self):
        img_labels = []
        with open(self.label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_file, label = line.strip().split()
                label = int(label)
                if label in LABEL_MAPPING:
                    img_labels.append((img_file, LABEL_MAPPING[label]))  # 将标签映射为 0-7 之间的值
        return img_labels

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name, label = self.img_labels[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_dataloader(img_dir, label_file, batch_size=32, split=0.8):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小为 224x224
        transforms.ToTensor(),
        transforms.Normalize([0.470, 0.442, 0.440], [0.206, 0.206, 0.198])  # 使用你的均值和方差
    ])
    
    dataset = SteelDataset(img_dir, label_file, transform=transform)
    
    train_size = int(split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
