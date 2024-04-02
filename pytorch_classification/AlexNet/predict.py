import os
import json
import intel_npu_acceleration_library
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import AlexNet


def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image_name = "IMG_20240331_162547.jpg"
    img_path = os.path.join(os.getcwd(), "predict_images", image_name)
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    json_path = "./class_indices.json"
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    model = AlexNet(num_classes=5).to(device)
    model = intel_npu_acceleration_library.compile(model, torch.float32, training=True)

    weights_path = "./AlexNet_2.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        pre = torch.softmax(output, dim=0)
        pre_cla = torch.argmax(pre).numpy()
    print_res = f"class: {class_indict[str(pre_cla)]} prob: {(pre[pre_cla].numpy() * 100):.2f}%"
    plt.title(print_res)
    for i in range(len(pre)):
        print(f"class {class_indict[str(i)]:10} prob: {pre[i].numpy() * 100:.2f}%")
    save_path = os.path.join(os.getcwd(), "predict_result", image_name)
    print(save_path)
    plt.savefig(save_path)


if __name__ == '__main__':
    predict()
