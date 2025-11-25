# tests/run_train_and_eval.py

import os
import sys

# -----------------------------
# Make project root importable
# -----------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar

from src.Data_Loading import DroneDataModule
from src.models.distance_lit_module import DistanceLitModule

# Optional custom callback (if you implemented it)
try:
    from src.Metrics import MetricsPlotter
except ImportError:
    MetricsPlotter = None


# Same log dir you used before
LOG_DIR = "/mnt/active_storage/Priontu/tmp/logs-lrddv3-distance-estimation"

# Same data roots you showed earlier
ROOT_IMAGES = r"/mnt/archive/LRDDv3"
ROOT_METADATA = r"/mnt/archive/LRDDv3/metadata"


def main():
    # -----------------------------
    # 1. DataModule (train/val/test)
    # -----------------------------
    batch_size = 4
    num_workers = 8
    model = "resnet50"
    img_width = 1280
    img_height = 1280
    hidden_layer_size = 64
    bbox_feat_dim = 4
    lr = 1e-4
    weight_decay = 1e-4
    patience = 5
    min_delta = 1e-4
    monitor = "val_mse"
    mode = "min"
    save_top_k=1
    save_last = True
    precision = "16-mixed"
    max_epochs = 20
    devices = 1
    log_every_n_steps = 10
    accumulate_grad_batches = 4
    

    dm = DroneDataModule(
        root_images=ROOT_IMAGES,
        root_metadata=ROOT_METADATA,
        batch_size=batch_size,
        num_workers=num_workers,
        img_width = img_width,
        img_height = img_height
        
    )

    # -----------------------------
    # 2. Model
    # -----------------------------
    model = DistanceLitModule(
        backbone_name=model,
        bbox_feat_dim=bbox_feat_dim,
        lr=lr,
        weight_decay=weight_decay,
        save_predictions=True,                 # <- for test
        predictions_path="test_predictions.csv",
        hidden_layer_size=hidden_layer_size
    )

    # -----------------------------
    # 3. Callbacks (ckpt, early stop, progress, metrics)
    # -----------------------------
    callbacks = [RichProgressBar(leave=True)]

    ckpt_dir = os.path.join(LOG_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="distpred-{epoch:02d}-{val_mse:.4f}",
        monitor=monitor,
        mode=mode,
        save_top_k=save_top_k,
        save_last=save_last,
    )
    callbacks.append(checkpoint_cb)

    early_stop_cb = EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=patience,
        min_delta=min_delta,
        verbose=True,
    )
    callbacks.append(early_stop_cb)

    if MetricsPlotter is not None:
        metrics_plotter = MetricsPlotter(log_dir=LOG_DIR)
        callbacks.append(metrics_plotter)

    # -----------------------------
    # 4. Trainer
    # -----------------------------
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        default_root_dir=LOG_DIR,
        callbacks=callbacks,
        log_every_n_steps=log_every_n_steps,
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches
    )

    # -----------------------------
    # 5. TRAIN
    # -----------------------------
    print("==== Training ====")
    trainer.fit(model, datamodule=dm)
    print(f"Best checkpoint: {checkpoint_cb.best_model_path}")
    print(f"Last checkpoint: {checkpoint_cb.last_model_path}")

    # -----------------------------
    # 6. EVALUATE (TEST) IN SAME SCRIPT
    # -----------------------------
    # Use the best checkpoint (by val_mse)
    best_ckpt_path = checkpoint_cb.best_model_path

    print("\n==== Loading best checkpoint for test ====")
    best_model = DistanceLitModule.load_from_checkpoint(
        best_ckpt_path,
        # backbone_name=model,
        # bbox_feat_dim=bbox_feat_dim,
        # lr=lr,
        # weight_decay=weight_decay,
        save_predictions=True,
        predictions_path="test_predictions.csv",
    )

    print("\n==== Testing on held-out test split ====")
    test_results = trainer.test(best_model, datamodule=dm)

    # Access numpy preds / targets / metrics from the LightningModule
    preds = best_model.test_preds_np
    targets = best_model.test_targets_np
    metrics = best_model.test_metrics

    print("\nTest loop finished.")
    print("Lightning test_results dict from trainer.test():")
    print(test_results)
    print("\nAggregated metrics from model.test_metrics:")
    print(metrics)

    # preds & targets are numpy arrays; you can further analyze or save them here
    # (predictions CSV is already written if save_predictions=True)


if __name__ == "__main__":
    main()
