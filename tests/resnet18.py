import os
from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights



log_dir = "/mnt/active_storage/Priontu/tmp/logs-lrddv3-distance-estimation"

class CSVImageDataset(Dataset):
    def __init__(self, images_root, metadata_root, transform=None):
        """
        images_root:   Path to split images root, e.g. LRDDv3/test
        metadata_root: Path to split metadata root, e.g. LRDDv3/metadata/test
        """
        self.images_root = Path(images_root)
        self.metadata_root = Path(metadata_root)
        self.transform = transform

        self.samples = []  # list of (image_path, distance)

        csv_files = sorted(self.metadata_root.glob("*.csv"))
        if not csv_files:
            raise RuntimeError(f"No CSVs found in {self.metadata_root}")

        for csv_path in csv_files:
            # csv name pattern: 04-11-2025_DJI_0007_metadata.csv
            stem = csv_path.stem              # "04-11-2025_DJI_0007_metadata"
            base = stem.replace("_metadata", "")  # "04-11-2025_DJI_0007"
            date_folder, clip_folder = base.split("_", 1)  # ("04-11-2025", "DJI_0007")

            images_dir = self.images_root / date_folder / clip_folder / "images"

            df = pd.read_csv(csv_path)

            # basic sanity check
            if "img_name" not in df.columns or "distance_3d_ft" not in df.columns:
                raise RuntimeError(f"Expected 'img_name' and 'distance_3d_ft' in {csv_path}")

            for _, row in df.iterrows():
                frame_name = row["img_name"]
                distance = float(row["distance_3d_ft"])
                img_path = images_dir / frame_name

                if not img_path.is_file():
                    # if you want to hard-fail instead, replace 'continue' with an exception
                    print(f"Warning: image not found, skipping: {img_path}")
                    continue

                self.samples.append((img_path, distance))

        if not self.samples:
            raise RuntimeError("No samples found — check paths and CSV contents.")

        print(f"Loaded {len(self.samples)} samples from {self.images_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, distance = self.samples[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        # regression target as float tensor
        y = torch.tensor(distance, dtype=torch.float32)

        return img, y


class DroneDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_images,
        root_metadata,
        batch_size=32,
        num_workers=4,
        img_size=224,
    ):
        """
        root_images:   path to LRDDv3 (contains 'test', 'validation')
        root_metadata: path to LRDDv3/metadata (contains 'test', 'validation')
        """
        super().__init__()
        self.root_images = Path(root_images)
        self.root_metadata = Path(root_metadata)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size

        # basic transforms; add augmentations here if you want
        self.train_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        self.val_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def setup(self, stage=None):
        # use 'test' split as training data
        if stage == "fit" or stage is None:
            self.train_ds = CSVImageDataset(
                images_root=self.root_images / "test",
                metadata_root=self.root_metadata / "test",
                transform=self.train_transform,
            )

            self.val_ds = CSVImageDataset(
                images_root=self.root_images / "val",
                metadata_root=self.root_metadata / "val",
                transform=self.val_transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class DistanceRegressor(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_features = backbone.fc.in_features
        backbone.fc = nn.Linear(num_features, 1)  # output: scalar distance

        self.model = backbone
        self.lr = lr

    def forward(self, x):
        return self.model(x).squeeze(1)  # (B,)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = F.mse_loss(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = F.mse_loss(preds, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    # TODO: change these to your actual paths
    # ROOT_IMAGES = r"/mnt/researchfiles/ECE IMAPLE/cluster_data/archive/LRDDv3"      # Imaple1
    # ROOT_METADATA = r"/mnt/researchfiles/ECE IMAPLE/cluster_data/archive/LRDDv3/metadata"       #Imaple1
    
    ROOT_IMAGES = r"/mnt/archive/LRDDv3"        # imaple 4
    ROOT_METADATA = r"/mnt/archive/LRDDv3/metadata"     # imaple 4

    dm = DroneDataModule(
        root_images=ROOT_IMAGES,
        root_metadata=ROOT_METADATA,
        batch_size=64,
        num_workers=8,
        img_size=224,
    )

    model = DistanceRegressor(lr=1e-3)

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        default_root_dir=log_dir
    )

    trainer.fit(model, datamodule=dm)
