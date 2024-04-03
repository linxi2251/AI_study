import os
import sys
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from model import AlexNet

root = os.path.abspath(os.path.join(os.getcwd(), "./"))  # get data root path


def load_dataset():
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }
    image_path = os.path.join(root, "dataset", "flower_data")
    assert os.path.exists(image_path), f"{image_path} path does not exist."
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"]
                                         )
    train_num = len(train_dataset)
    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print(f"Using {nw} data_loader workers every process")

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=nw)
    validate_dateset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dateset)
    validate_loader = DataLoader(validate_dateset,
                                 batch_size=4, shuffle=False,
                                 num_workers=nw)
    print(f"using {train_num} images for training, {val_num} images for validation.")
    return train_loader, validate_loader, train_num, val_num
    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.__next__()
    # image_show(utils.make_grid(test_image))


def image_show(img):
    img = img / 2 + 0.5  # un_normalize
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    print(os.path.join(root, "tmp"))
    plt.savefig(os.path.join(root, "tmp", "image.png"))


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    train_loader, validate_loader, train_num, val_num = load_dataset()
    net = AlexNet(num_classes=5, init_weights=True)
    net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    epochs = 500
    save_path = "./AlexNet.pth"
    best_acc = 0.0
    train_steps = len(train_loader)
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for images, labels in train_bar:
            # images = images.to(device)
            # labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_fn(outputs, labels)
            loss.backward()   # 反向传播
            optimizer.step()  # 更新每个节点的参数

            running_loss += loss.item()
            train_bar.desc = f"train epoch[{epoch + 1}/{epochs}] loss:{loss:.3f}"
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                # val_images, val_labels = val_images.to(device), val_labels.to(device)
                outputs = net(val_images)
                pre = torch.max(outputs, dim=1)[1]
                acc += torch.eq(pre, val_labels).sum().item()
        val_acc = acc / val_num
        print(f"[epoch {epoch + 1}] train_loss: {(running_loss / train_steps):.3f} val_accuracy: {val_acc}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)
    print("Finished Training")


if __name__ == '__main__':
    train()
