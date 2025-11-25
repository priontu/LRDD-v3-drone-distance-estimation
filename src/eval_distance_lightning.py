# src/eval_distance_lightning.py

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, parent_dir)

import torch
import pytorch_lightning as pl

from src.Data_Loading import DroneDataModule
from src.models.distance_lit_module import DistanceLitModule


def evaluate_model_lightning(
    checkpoint_path,
    backbone="resnet50",
    batch_size=16,
    device="cuda",
    save_predictions=True,
    bbox_feature_dim=4,
    num_workers=8,
    predictions_path="test_predictions.csv",
):
    ROOT_IMAGES = r"/mnt/archive/LRDDv3"
    ROOT_METADATA = r"/mnt/archive/LRDDv3/metadata"

    dm = DroneDataModule(
        root_images=ROOT_IMAGES,
        root_metadata=ROOT_METADATA,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # We only need test set
    dm.setup(stage="test")

    lit_model = DistanceLitModule.load_from_checkpoint(
        checkpoint_path,
        backbone_name=backbone,
        bbox_feat_dim=bbox_feature_dim,
        lr=1e-4,
        weight_decay=1e-4,
        save_predictions=save_predictions,
        predictions_path=predictions_path,
    )

    accelerator = "gpu" if device.startswith("cuda") and torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        log_every_n_steps=10,
    )

    trainer.test(lit_model, datamodule=dm)

    preds = lit_model.test_preds_np
    targets = lit_model.test_targets_np
    metrics = lit_model.test_metrics

    print("\nFinal test metrics:", metrics)
    return preds, targets, metrics


if __name__ == "__main__":
    # Example usage: update checkpoint_path as needed
    preds, targets, metrics = evaluate_model_lightning(
        checkpoint_path="/path/to/checkpoints/distpred-epoch.ckpt",
        predictions_path="test_predictions.csv",
    )
