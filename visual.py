import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from model import CNNClassifier
from data_loader import create_dataloader

# 将分类结果从 0-7 映射回 [-30, -25, -20, -15, -10, -5, 0, 5]
REVERSE_LABEL_MAPPING = {0: -30, 1: -25, 2: -20, 3: -15, 4: -10, 5: -5, 6: 0, 7: 5}

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('teaser/confusion_matrix.png')
    plt.show()

def visualize_tsne(features, labels):
    tsne = TSNE(n_components=2, random_state=0)
    
    # 将 features 转换为 NumPy 数组
    features = np.array(features)
    
    # 执行 t-SNE
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='jet', alpha=0.6)
    plt.legend(*scatter.legend_elements(), title='Classes', loc='upper right')
    plt.title('t-SNE Visualization')
    plt.savefig('teaser/tsne_visualization.png')
    plt.show()

def test_and_visualize(img_dir, label_file, batch_size=32):
    _, test_loader = create_dataloader(img_dir, label_file, batch_size=batch_size)

    model = CNNClassifier(num_classes=8)
    model.load_state_dict(torch.load('cnn_classifier.pth'))
    model = model.cuda()
    model.eval()

    all_labels = []
    all_predictions = []
    all_features = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_features.extend(outputs.cpu().numpy())  # 获取模型输出用于 t-SNE

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)

    # 绘制并保存混淆矩阵
    classes = list(REVERSE_LABEL_MAPPING.values())  # 获取类别名称
    plot_confusion_matrix(cm, classes)

    # 可视化 t-SNE
    visualize_tsne(all_features, all_predictions)

if __name__ == "__main__":
    test_and_visualize('Angle20200610/img-20200610', 'Angle20200610/angle_label_20200610.txt')
