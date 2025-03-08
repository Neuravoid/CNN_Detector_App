import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import os
import gc  # 🛠 Bellek Temizleme için

# 🎯 Modeli Yükleme Fonksiyonu (Lazy Load)
def load_model():
    global model
    if "model" not in globals():
        class ResNet50Classifier(nn.Module):
            def __init__(self):
                super(ResNet50Classifier, self).__init__()
                self.model = models.resnet50(weights=None)
                for param in list(self.model.parameters())[:-10]:
                    param.requires_grad = False
                num_ftrs = self.model.fc.in_features
                self.model.fc = nn.Sequential(
                    nn.Linear(num_ftrs, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 1)
                )
            def forward(self, x):
                return self.model(x)

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, "..", "model", "final_model.pth")
        model = ResNet50Classifier()
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")), strict=False)
        model.eval()
    return model

# 📸 Görüntü İşleme
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 🚀 FastAPI Sunucusu
app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image = transform(image).unsqueeze(0)

        model = load_model()  # 🛠 Lazy Load Model
        with torch.no_grad():
            output = model(image)
            confidence = torch.sigmoid(output).item()

        result = "Human" if confidence > 0.5 else "Not Human"

    except Exception as e:
        return {"error": str(e)}

    finally:
        # 🚀 RAM Temizleme İşlemi
        del image
        del output
        del model  # 🛠 Model nesnesini silerek RAM tüketimini azalt
        gc.collect()  # Belleği temizle

    return {"prediction": result, "confidence": confidence}
