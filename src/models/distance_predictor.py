# src/models/distance_predictor.py

import torch
import torch.nn as nn
import torchvision.models as models


class DistancePredictor(nn.Module):
    def __init__(self, backbone_name="resnet18", bbox_feat_dim=4, hidden_layer_size = 128):
        super().__init__()

        if backbone_name == "resnet18":
            self.backbone_full = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.backbone_crop = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            backbone_out_dim = 512
        elif backbone_name == "resnet50":
            self.backbone_full = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.backbone_crop = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            backbone_out_dim = 2048
        elif backbone_name == "resnet101":
            self.backbone_full = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            self.backbone_crop = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            backbone_out_dim = 2048
        elif backbone_name == "resnet152":
            self.backbone_full = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
            self.backbone_crop = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
            backbone_out_dim = 2048
        else:
            raise ValueError(f"Invalid backbone: {backbone_name}")

        # Remove final classification layer
        self.backbone_full.fc = nn.Identity()
        self.backbone_crop.fc = nn.Identity()

        combined_dim = backbone_out_dim * 2 + bbox_feat_dim

        self.head = nn.Sequential(
            nn.Linear(combined_dim, hidden_layer_size),
            nn.Dropout(0.2),
            nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            
            nn.Linear(hidden_layer_size, 1),
        )

    def forward(self, full_img, crop_img, bbox_features):
        full_feat = self.backbone_full(full_img)
        crop_feat = self.backbone_crop(crop_img)
        x = torch.cat([full_feat, crop_feat, bbox_features], dim=1)
        out = self.head(x)
        return out.squeeze(1)
