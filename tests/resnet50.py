import os
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
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt


log_dir = "/mnt/active_storage/Priontu/tmp/logs-lrddv3-distance-estimation"

class LossHistory(pl.Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        # Depending on PL version, it might be "train_loss" or "train_loss_epoch"
        train_loss = metrics.get("train_loss") or metrics.get("train_loss_epoch")
        if train_loss is not None:
            self.train_losses.append(train_loss.detach().cpu().item())

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        val_loss = metrics.get("val_loss") or metrics.get("val_loss_epoch")
        if val_loss is not None:
            self.val_losses.append(val_loss.detach().cpu().item())

    def on_fit_end(self, trainer, pl_module):
        # Called once training is finished
        if not self.train_losses and not self.val_losses:
            print("LossHistory: no losses recorded, skipping plot.")
            return

        os.makedirs(self.log_dir, exist_ok=True)
        epochs = range(1, len(self.train_losses) + 1)

        plt.figure()
        if self.train_losses:
            plt.plot(epochs, self.train_losses, label="train_loss")
        if self.val_losses:
            # Make sure lengths align; truncate to min length just in case
            n = min(len(self.val_losses), len(epochs))
            plt.plot(epochs[:n], self.val_losses[:n], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()

        out_path = os.path.join(self.log_dir, "loss_curves.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"LossHistory: saved loss curves to {out_path}")


class MetricsPlotter(pl.Callback):
    """
    Collects specified metrics over epochs and plots train/val curves at the end.
    """
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir

        # history[metric_name] = [values per epoch]
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_mae": [],
            "val_mae": [],
            "train_rmse": [],
            "val_rmse": [],
            "train_accuracy_tol": [],
            "val_accuracy_tol": [],
        }

        # For convenience: what to plot together (train vs val)
        self.metric_pairs = [
            ("train_loss", "val_loss"),
            ("train_mae", "val_mae"),
            ("train_rmse", "val_rmse"),
            ("train_accuracy_tol", "val_accuracy_tol"),
        ]

    def _get_metric(self, metrics, name):
        """
        Helper: try direct name, then name + '_epoch' (PL version differences).
        Returns python float or None.
        """
        value = metrics.get(name)
        if value is None:
            value = metrics.get(f"{name}_epoch")
        if value is None:
            return None

        # tensor -> float
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().item()
        return float(value)

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        for key in list(self.history.keys()):
            if key.startswith("train_"):
                val = self._get_metric(metrics, key)
                if val is not None:
                    self.history[key].append(val)

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        for key in list(self.history.keys()):
            if key.startswith("val_"):
                val = self._get_metric(metrics, key)
                if val is not None:
                    self.history[key].append(val)

    def on_fit_end(self, trainer, pl_module):
        os.makedirs(self.log_dir, exist_ok=True)

        for train_key, val_key in self.metric_pairs:
            train_vals = self.history[train_key]
            val_vals = self.history[val_key]

            if not train_vals and not val_vals:
                # Nothing recorded for this metric, skip
                continue

            # Align lengths just in case one is shorter
            n_epochs = min(len(train_vals), len(val_vals)) if (train_vals and val_vals) else max(len(train_vals), len(val_vals))
            epochs = list(range(1, n_epochs + 1))

            plt.figure()
            if train_vals:
                plt.plot(epochs[:len(train_vals)], train_vals[:len(epochs)], label=train_key)
            if val_vals:
                plt.plot(epochs[:len(val_vals)], val_vals[:len(epochs)], label=val_key)

            plt.xlabel("Epoch")
            plt.ylabel(train_key.replace("train_", "").replace("_", " ").upper())
            plt.title(f"{train_key} / {val_key}")
            plt.legend()
            plt.grid(True)

            filename = f"{train_key}_and_{val_key}.png"
            out_path = os.path.join(self.log_dir, filename)
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()

            print(f"MetricsPlotter: saved {filename} to {self.log_dir}")


class CSVImageDataset(Dataset):
    def __init__(self, images_root, metadata_root, transform=None):
        self.images_root = Path(images_root)
        self.metadata_root = Path(metadata_root)
        self.transform = transform

        self.samples = []

        csv_files = sorted(self.metadata_root.glob("*.csv"))
        if not csv_files:
            raise RuntimeError(f"No CSVs found in {self.metadata_root}")

        for csv_path in csv_files:
            stem = csv_path.stem
            base = stem.replace("_metadata", "")
            date_folder, clip_folder = base.split("_", 1)

            images_dir = self.images_root / date_folder / clip_folder / "images"
            df = pd.read_csv(csv_path)

            if "img_name" not in df.columns or "distance_3d_ft" not in df.columns:
                raise RuntimeError(f"Expected 'img_name' and 'distance_3d_ft' in {csv_path}")

            for _, row in df.iterrows():
                frame_name = row["img_name"]
                raw_distance = row["distance_3d_ft"]

                # ---- NEW: robust distance parsing ----
                try:
                    distance = float(raw_distance)
                except Exception:
                    print(f"Warning: bad distance value {raw_distance!r} in {csv_path}, skipping")
                    continue

                if not math.isfinite(distance):
                    print(f"Warning: non-finite distance {distance} in {csv_path}, skipping")
                    continue
                # --------------------------------------

                img_path = images_dir / frame_name
                if not img_path.is_file():
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
    def __init__(self, lr=1e-3, tolerance=5.0):
        super().__init__()
        self.save_hyperparameters()

        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
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
        callbacks=[metrics_plotter],
    )

    trainer.fit(model, datamodule=dm)
