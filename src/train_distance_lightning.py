# src/train_distance_lightning.py

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, parent_dir)

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar

from src.Data_Loading import DroneDataModule
from src.models.distance_lit_module import DistanceLitModule

# from your previous script:
log_dir = "/mnt/active_storage/Priontu/tmp/logs-lrddv3-distance-estimation"

# Optional: your custom metrics plotter, if you have it
try:
    from src.Metrics import MetricsPlotter
except ImportError:
    MetricsPlotter = None


def main():
    # copied from your previous code
    ROOT_IMAGES = r"/mnt/archive/LRDDv3"
    ROOT_METADATA = r"/mnt/archive/LRDDv3/metadata"

    batch_size = 4
    num_workers = 8

    dm = DroneDataModule(
        root_images=ROOT_IMAGES,
        root_metadata=ROOT_METADATA,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = DistanceLitModule(
        backbone_name="resnet50",
        bbox_feat_dim=4,
        lr=1e-4,
        weight_decay=1e-4,
        save_predictions=False,
    )

    callbacks = [RichProgressBar()]

    # a simple checkpoint callback (best by val_mse)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="distpred-{epoch:02d}-{val_mse:.4f}",
        monitor="val_mse",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    callbacks.append(ckpt_cb)

    # early stopping (optional)
    early_stop_cb = EarlyStopping(
        monitor="val_mse",
        mode="min",
        patience=5,
        min_delta=1e-4,
        verbose=True,
    )
    callbacks.append(early_stop_cb)

    # your MetricsPlotter, if available
    if MetricsPlotter is not None:
        metrics_plotter = MetricsPlotter(log_dir=log_dir)
        callbacks.append(metrics_plotter)

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        default_root_dir=log_dir,
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=dm)
    print(f"Best checkpoint: {ckpt_cb.best_model_path}")


if __name__ == "__main__":
    main()
