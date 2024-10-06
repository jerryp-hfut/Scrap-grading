import torch
from model import CNNClassifier
from data_loader import create_dataloader
from tqdm import tqdm

# 将 0-7 的预测结果映射回原始标签值
REVERSE_LABEL_MAPPING = {0: -30, 1: -25, 2: -20, 3: -15, 4: -10, 5: -5, 6: 0, 7: 5}

def test_model(img_dir, label_file, batch_size=32):
    _, test_loader = create_dataloader(img_dir, label_file, batch_size=batch_size)

    model = CNNClassifier(num_classes=8)
    model.load_state_dict(torch.load('cnn_classifier.pth'))
    model = model.cuda()
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as tepoch:
            tepoch.set_description("Testing")
            for images, labels in tepoch:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')

if __name__ == "__main__":
    test_model('Angle20200610/img-20200610', 'Angle20200610/angle_label_20200610.txt')
