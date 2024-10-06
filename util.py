import torch
from PIL import Image
import torchvision.transforms as transforms
from model import CNNClassifier
from tkinter import Tk, Label, Button, filedialog
from tkinter.messagebox import showinfo
from PIL import ImageTk

# 将分类结果从 0-7 映射回 [-30, -25, -20, -15, -10, -5, 0, 5]
REVERSE_LABEL_MAPPING = {0: -30, 1: -25, 2: -20, 3: -15, 4: -10, 5: -5, 6: 0, 7: 5}

# 图像分类预测函数
def predict_rating(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor(),
        transforms.Normalize([0.470, 0.442, 0.440], [0.206, 0.206, 0.198])  # 标准化
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 增加 batch 维度
    
    model = CNNClassifier(num_classes=8)
    model.load_state_dict(torch.load('cnn_classifier.pth'))
    model = model.cuda()
    model.eval()
    
    with torch.no_grad():
        image = image.cuda()
        output = model(image)
        _, predicted = torch.max(output, 1)
        rating = REVERSE_LABEL_MAPPING[predicted.item()]  # 将预测结果映射回原始标签值
        
        return rating

# 创建 Tkinter 界面
class ImageClassifierApp:
    def __init__(self, master):
        self.master = master
        master.title("Image Rating Predictor")

        self.label = Label(master, text="请选择一张图片进行分类预测")
        self.label.pack(pady=20)

        self.upload_button = Button(master, text="选择图片", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.image_label = Label(master)
        self.image_label.pack(pady=20)

        self.result_label = Label(master, text="")
        self.result_label.pack(pady=10)

    def upload_image(self):
        # 选择文件对话框
        file_path = filedialog.askopenfilename()
        if file_path:
            # 读取并显示图片
            image = Image.open(file_path)
            image.thumbnail((300, 300))  # 调整显示的图片大小
            img = ImageTk.PhotoImage(image)
            self.image_label.configure(image=img)
            self.image_label.image = img

            # 调用模型预测
            rating = predict_rating(file_path)
            self.result_label.config(text=f"预测结果: {rating}")

# 启动应用
if __name__ == "__main__":
    root = Tk()
    app = ImageClassifierApp(root)
    root.mainloop()
