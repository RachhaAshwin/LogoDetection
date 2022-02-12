from torchvision import models, transforms
import torch
import torch.nn as nn
from PIL import Image
import gdown
cloud_model_location = "1O5tAg5I2wlBynGkEfHWPWmyTSIUlubhy"
#from GD_download import download_file_from_google_drive
url = 'https://drive.google.com/uc?id=1O5tAg5I2wlBynGkEfHWPWmyTSIUlubhy'
def predict(image_path):
    resnet = models.resnet34(pretrained=False)
    state_dict = torch.load('resnet.pth', map_location=torch.device('cpu'))
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 18)
    resnet.load_state_dict(state_dict)
    #https://pytorch.org/docs/stable/torchvision/models.html
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )])

    img = Image.open(image_path)
    batch_t = torch.unsqueeze(transform(img), 0)

    resnet.eval()
    out = resnet(batch_t)

    classes = ['Acura',
 'Audi',
 'BMW',
 'Chevrolet',
 'Ford',
 'Honda',
 'Hyundai',
 'Infiniti',
 'KIA',
 'Lamborghini',
 'Lexus',
 'Mazda',
 'MercedesBenz',
 'Nissan',
 'Porsche',
 'Tesla',
 'Toyota',
 'Volkswagen']

    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]
