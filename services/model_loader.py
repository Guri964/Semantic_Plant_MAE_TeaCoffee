import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ViT_B_16_Weights
from config import CROP_MODEL_PATH, HEALTH_MODEL_PATH, CROP_CLASSES, HEALTH_CLASSES


class CropModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CropModel, self).__init__()
        self.backbone = models.efficientnet_b4(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


class HealthModel(nn.Module):
    def __init__(self, num_classes=2):
        super(HealthModel, self).__init__()
        self.efficientnet = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
        )
        self.efficientnet.classifier = nn.Identity()
        self.eff_out_features = 1792

        self.vit = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads = nn.Identity()
        self.vit_out_features = 768

        combined = self.eff_out_features + self.vit_out_features

        self.fusion = nn.Sequential(
            nn.Linear(combined, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        eff_features = self.efficientnet(x)
        vit_features = self.vit(x)
        combined     = torch.cat([eff_features, vit_features], dim=1)
        out          = self.fusion(combined)
        return out


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_models():
    device = get_device()

    crop_model = CropModel(num_classes=len(CROP_CLASSES))
    crop_model.load_state_dict(
        torch.load(CROP_MODEL_PATH, map_location=device)
    )
    crop_model.to(device)
    crop_model.eval()

    health_model = HealthModel(num_classes=len(HEALTH_CLASSES))
    health_model.load_state_dict(
        torch.load(HEALTH_MODEL_PATH, map_location=device)
    )
    health_model.to(device)
    health_model.eval()

    return crop_model, health_model, device