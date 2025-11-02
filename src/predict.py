import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import sys

image_path = sys.argv[1]

# MODELİ TEKRAR OLUŞTUR
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

# AĞIRLIKLARI YÜKLE
model.load_state_dict(torch.load("models/chest_xray_model.pth", map_location=torch.device("cpu")))
model.eval()

# GÖRÜNTÜ DÖNÜŞÜMÜ
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# GÖRÜNTÜYÜ YÜKLE
img = Image.open(image_path).convert("RGB")
img_t = transform(img).unsqueeze(0)

# TAHMİN
with torch.no_grad():
    outputs = model(img_t)
    _, predicted = torch.max(outputs, 1)

classes = ["NORMAL", "PNEUMONIA"]
print("Prediction:", classes[predicted.item()])

