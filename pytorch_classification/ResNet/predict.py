import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import resnet34

images_root_dir = os.path.join(os.getcwd(), "../../images")
def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image_name = input("input a image name.\n")
    img_path = os.path.join(images_root_dir, "predict_images", image_name)
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    json_path = "./class_indices.json"
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    model = resnet34(num_classes=5)
    model = model.to(device)

    weights_path = "./resnet34.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        pre = torch.softmax(output, dim=0)
        pre_cla = torch.argmax(pre).numpy()
    print_res = f"class: {class_indict[str(pre_cla)]} prob: {(pre[pre_cla].numpy() * 100):.2f}%"
    plt.title(print_res)
    for i in range(len(pre)):
        print(f"class {class_indict[str(i)]:10} prob: {pre[i].numpy() * 100:.2f}%")
    save_path = os.path.join(images_root_dir, "predict_result", image_name)
    print(save_path)
    plt.savefig(save_path)


if __name__ == '__main__':
    predict()
