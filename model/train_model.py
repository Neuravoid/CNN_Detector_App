import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import BinaryAccuracy

# **RGB'ye çevirme fonksiyonu**
def convert_to_rgb(img):
    return img.convert('RGB')


# **ResNet50 Modeli**
class ResNet50Classifier(pl.LightningModule):
    def __init__(self, num_classes=1):
        super(ResNet50Classifier, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Son 10 katmanı serbest bırak, geri kalanı dondur
        for param in list(self.model.parameters())[:-10]:
            param.requires_grad = False

        # Yeni FC katmanı (Binary Classification)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)  # Sigmoid kullanmıyoruz!
        )

        self.criterion = nn.BCEWithLogitsLoss()  # Sigmoid yerine Logits kullanılıyor
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        labels = labels.float().unsqueeze(1)
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = self.train_acc(torch.sigmoid(outputs), labels.int())  # Sigmoid burada uygulanıyor
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        labels = labels.float().unsqueeze(1)
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = self.val_acc(torch.sigmoid(outputs), labels.int())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }


# **Eğitim sürecini başlat**
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    torch.multiprocessing.set_start_method('spawn', force=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")

    # **Transform Ayarları (512x512)**
    train_transforms = transforms.Compose([
        transforms.Lambda(convert_to_rgb),
        transforms.RandomResizedCrop((512, 512), scale=(0.8, 1.0)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Lambda(convert_to_rgb),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # **Veri Yükleme**
    train_data = datasets.ImageFolder(root="C:/Users/hp/PycharmProjects/cnn_project/data/h_nh/training_set",
                                      transform=train_transforms)
    val_data = datasets.ImageFolder(root="C:/Users/hp/PycharmProjects/cnn_project/data/h_nh/test_set",
                                    transform=val_transforms)

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False, num_workers=4)

    model = ResNet50Classifier()

    trainer = pl.Trainer(max_epochs=5, accelerator="gpu" if torch.cuda.is_available() else "cpu")
    trainer.fit(model, train_loader, val_loader)

    # **Modeli .pth formatında kaydet**
    model_path = "//model/final_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model kaydedildi: {model_path}")