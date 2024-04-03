import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from model import LeNet
from torchinfo import summary
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore", message="Applied workaround for CuDNN issue")

# 预处理图像
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# 加载数据集
train_dataset = torchvision.datasets.CIFAR10(
    root="../../data_set", train=True, download=False, transform=transform)
train_len = len(train_dataset)

val_dataset = torchvision.datasets.CIFAR10(
    root="../../data_set", train=False, download=False, transform=transform)
val_num = len(val_dataset)
# 加载dataloader
train_loader = DataLoader(train_dataset, batch_size=32,
                          shuffle=True, num_workers=8)

val_loader = DataLoader(val_dataset, batch_size=10000,
                          shuffle=True, num_workers=8)

# 加载设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
torch.backends.cudnn.benchmark = True
# 实例化模型
model = LeNet()
model.to(device)
summary(model)

# 设置损失函数
loss_function = nn.CrossEntropyLoss()
# 设置优化器
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 训练epochs
epochs = 20
best_acc = 0.0
train_steps = len(train_loader)

best_acc = 0.0
model_dic = {}
for epoch in range(epochs):
    model.train()
    train_bar = tqdm(train_loader)
    running_loss = 0.0
    strain_steps = len(train_loader)
    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        train_bar.desc = desc=f"train epoch[{epoch+1}/{epochs}] loss:{loss:.3f}"

    # val
    model.eval()
    acc = 0.0
    with torch.no_grad():
        val_bar = tqdm(val_loader)
        for images, labels in val_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, labels).sum().item()
    val_accurate = acc / val_num
    print(f"[epoch {epoch+1}] train_avg_loss:{(running_loss / strain_steps):.3f} val_acc:{val_accurate * 100:.2f}%")
    if val_accurate > best_acc:
        best_acc = val_accurate
        model_dic = model.state_dict()

print("Finished Done")
torch.save(model_dic, f"./LeNet_{best_acc*100:.2f}%.pth")
print("")
print(f"best accuracy:{best_acc * 100:.2f}%")

