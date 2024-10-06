import torch
import torch.optim as optim
import torch.nn as nn
from model import CNNClassifier
from data_loader import create_dataloader
from tqdm import tqdm  # 用于显示进度条
import os
def train_model(img_dir, label_file, num_epochs=10, learning_rate=0.001, batch_size=32):
    train_loader, test_loader = create_dataloader(img_dir, label_file, batch_size=batch_size)
    
    model = CNNClassifier(num_classes=8)  # 分类任务有8个类别
    model = model.cuda()  # 如果有GPU可以使用.cuda()
    if os.path.exists("cnn_classifier.pth"):
        model.load_state_dict(torch.load("cnn_classifier.pth"))
        print("最新模型参数成功加载。")
    else:
        print("未找到参数，重新训练。")
    criterion = nn.CrossEntropyLoss()  # 分类任务的损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            for images, labels in tepoch:
                images, labels = images.cuda(), labels.cuda()
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), 'cnn_classifier.pth')
    print('Model training complete.')

if __name__ == "__main__":
    train_model('Angle20200610/img-20200610', 'Angle20200610/angle_label_20200610.txt')

    