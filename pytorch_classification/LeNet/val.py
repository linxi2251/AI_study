from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from model import LeNet
from torchinfo import summary
from tqdm import tqdm
import torch

import warnings

warnings.filterwarnings("ignore", message="Applied workaround for CuDNN issue")

# 预处理图像
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# 加载数据集
val_dataset = torchvision.datasets.CIFAR10(
    root="../../data_set", train=False, download=False, transform=transform)
val_num = len(val_dataset)
# 加载dataloader

val_loader = DataLoader(val_dataset, batch_size=10000,
                        shuffle=True, num_workers=8)

# 加载设备
device = torch.device("cpu")
print("using {} device.".format(device))
torch.backends.cudnn.benchmark = True
# 实例化模型
model = LeNet()
model.to(device)
summary(model)

model.load_state_dict(torch.load("./LeNet_65.30%.pth", map_location=device))
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
print(f"val_acc:{val_accurate * 100:.2f}%")
