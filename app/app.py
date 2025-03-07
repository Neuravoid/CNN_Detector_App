import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import os

# 🎯 Fine-Tuning Yapılmış Modeli Tanımla (train_model.py ile aynı yapı!)
class ResNet50Classifier(nn.Module):
    def __init__(self):
        super(ResNet50Classifier, self).__init__()
        self.model = models.resnet50(weights=None)  # Pretrained=False kullanıyoruz çünkü eğitilmiş modelin var!
        
        # Son 10 katmanı serbest bırak, geri kalanını dondur (Aynı şekilde eğittik!)
        for param in list(self.model.parameters())[:-10]:
            param.requires_grad = False

        # Yeni Fully Connected Katmanı (train_model.py ile aynı!)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)  # Binary Classification için çıktı 1 nöron
        )

    def forward(self, x):
        return self.model(x)


# 🔥 Eğitilmiş Modeli Yükle (final_model.pth)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "model", "final_model.pth")
model = ResNet50Classifier()

# Eğitilmiş ağırlıkları yükle
state_dict = torch.load(model_path, map_location=torch.device("cpu"))
model.load_state_dict(state_dict, strict=False)  # ✅ strict=False: Fazla layer hatası engellenir

model.eval()  # Modeli inference moduna al

# 📸 Görüntü İşleme (train sırasında kullandığın transformları koruyoruz)
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 🚀 FastAPI Sunucusu
app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        confidence = torch.sigmoid(output).item()

    print(f"Model Çıktısı (Logit): {output.item()} - Olasılık (Sigmoid): {confidence}")

    result = "Human" if confidence > 0.5 else "Not Human"

    return {"prediction": result, "confidence": confidence}