import os
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, parent_dir)

from pathlib import Path

import pandas as pd
from PIL import Image

import math
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
from src.Metrics import MetricsPlotter
from src.Data_Loading_old import DroneDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar


log_dir = "/mnt/active_storage/Priontu/tmp/logs-lrddv3-distance-estimation"

class DistanceRegressor(pl.LightningModule):
    def __init__(self, lr=1e-3, tolerance=5.0):
        super().__init__()
        self.save_hyperparameters()

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_features = backbone.fc.in_features
        backbone.fc = nn.Linear(num_features, 1)

        self.model = backbone
        self.lr = lr
        self.tolerance = tolerance   # accuracy tolerance in feet

    def forward(self, x):
        return self.model(x).squeeze(1)

    def _compute_metrics(self, preds, y):
        """Returns loss, mae, rmse, accuracy-with-tolerance."""
        loss = F.mse_loss(preds, y)
        mae = torch.mean(torch.abs(preds - y))
        rmse = torch.sqrt(loss)
        accuracy = (torch.abs(preds - y) < self.tolerance).float().mean()
        return loss, mae, rmse, accuracy

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss, mae, rmse, acc = self._compute_metrics(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_mae", mae, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_rmse", rmse, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_accuracy_tol", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss, mae, rmse, acc = self._compute_metrics(preds, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_mae", mae, on_epoch=True, prog_bar=True)
        self.log("val_rmse", rmse, on_epoch=True, prog_bar=True)
        self.log("val_accuracy_tol", acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)




if __name__ == "__main__":
    # TODO: change these to your actual paths
    # ROOT_IMAGES = r"/mnt/researchfiles/ECE IMAPLE/cluster_data/archive/LRDDv3"      # Imaple1
    # ROOT_METADATA = r"/mnt/researchfiles/ECE IMAPLE/cluster_data/archive/LRDDv3/metadata"       #Imaple1
    
    ROOT_IMAGES = r"/mnt/archive/LRDDv3"
    ROOT_METADATA = r"/mnt/archive/LRDDv3/metadata"

    dm = DroneDataModule(
        root_images=ROOT_IMAGES,
        root_metadata=ROOT_METADATA,
        batch_size=64,
        num_workers=8,
        img_size=224,
    )

    model = DistanceRegressor(lr=1e-3, tolerance=5.0)

    metrics_plotter = MetricsPlotter(log_dir=log_dir)

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        default_root_dir=log_dir,
        callbacks=[metrics_plotter, RichProgressBar()],
    )

    trainer.fit(model, datamodule=dm)
