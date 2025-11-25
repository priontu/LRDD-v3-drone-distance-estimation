# src/models/distance_lit_module.py

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from .distance_predictor import DistancePredictor


class DistanceLitModule(pl.LightningModule):
    def __init__(
        self,
        backbone_name: str,
        bbox_feat_dim: int = 4,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        save_predictions: bool = False,
        predictions_path: str = "test_predictions.csv",
        tolerance: float = 5.0,  # accuracy tolerance in feet,
        hidden_layer_size:int = 128
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = DistancePredictor(
            backbone_name=backbone_name,
            bbox_feat_dim=bbox_feat_dim,
            hidden_layer_size=hidden_layer_size
        )

        self._test_preds = []
        self._test_targets = []

        self.test_preds_np = None
        self.test_targets_np = None
        self.test_metrics = None
        
        self.tolerance = tolerance

    # ------------- helpers -------------

    def _compute_metrics(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Returns:
            loss (MSE), mse, rmse, mae, accuracy_with_tolerance
        """
        # make sure shapes match
        target = target.view_as(preds).float()

        mse = F.mse_loss(preds, target)
        mae = torch.mean(torch.abs(preds - target))
        rmse = torch.sqrt(mse)

        # accuracy within tolerance (in same units as distance_3d_ft)
        diff = torch.abs(preds - target)
        acc = (diff < self.tolerance).float().mean()

        # here "loss" is just mse; you could swap for something else if needed
        loss = mse
        return loss, mse, rmse, mae, acc

    def forward(self, full_img, crop_img, bbox_features):
        return self.model(full_img, crop_img, bbox_features)

    # -------- train --------
    def training_step(self, batch, batch_idx):
        full_img, crop_img, bbox_features, distance = batch
        preds = self(full_img, crop_img, bbox_features)

        loss, mse, rmse, mae, acc = self._compute_metrics(preds, distance)

        # log everything for training
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_mse", mse, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_mae", mae, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_rmse", rmse, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_accuracy_tol", acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    # -------- val --------
    def validation_step(self, batch, batch_idx):
        full_img, crop_img, bbox_features, distance = batch
        preds = self(full_img, crop_img, bbox_features)

        loss, mse, rmse, mae, acc = self._compute_metrics(preds, distance)

        # log everything for validation
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mse", mse, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_mae", mae, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_rmse", rmse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy_tol", acc, on_step=False, on_epoch=True, prog_bar=True)

    # -------- test --------
    def on_test_start(self):
        self._test_preds = []
        self._test_targets = []
        self.test_preds_np = None
        self.test_targets_np = None
        self.test_metrics = None

    def test_step(self, batch, batch_idx):
        full_img, crop_img, bbox_features, distance = batch
        preds = self(full_img, crop_img, bbox_features)

        loss, mse, rmse, mae, acc = self._compute_metrics(preds, distance)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_mse", mse, on_step=False, on_epoch=True)
        self.log("test_rmse", rmse, on_step=False, on_epoch=True)
        self.log("test_mae", mae, on_step=False, on_epoch=True)
        self.log("test_accuracy_tol", acc, on_step=False, on_epoch=True)

        self._test_preds.append(preds.detach().cpu())
        self._test_targets.append(distance.detach().cpu())

        return loss

    def on_test_epoch_end(self):
        if len(self._test_preds) == 0:
            return

        preds = torch.cat(self._test_preds).view(-1).numpy()
        targets = torch.cat(self._test_targets).view(-1).numpy()

        mse = float(np.mean((preds - targets) ** 2))
        mae = float(np.mean(np.abs(preds - targets)))
        rmse = float(np.sqrt(mse))

        ss_res = float(np.sum((preds - targets) ** 2))
        ss_tot = float(np.sum((targets - targets.mean()) ** 2)) if len(targets) > 1 else 0.0
        r2 = float(1.0 - ss_res / (ss_tot + 1e-8)) if ss_tot > 0 else 0.0

        # accuracy within tolerance at numpy level
        tol = self.hparams.tolerance
        acc = float(np.mean(np.abs(preds - targets) < tol))

        self.log("test_mse_final", mse)
        self.log("test_mae_final", mae)
        self.log("test_rmse_final", rmse)
        self.log("test_r2", r2)
        self.log("test_accuracy_tol_final", acc)

        print("\n===== Test Metrics =====")
        print(f"MAE      : {mae:.4f}")
        print(f"MSE      : {mse:.4f}")
        print(f"RMSE     : {rmse:.4f}")
        print(f"R²       : {r2:.4f}")
        print(f"ACC@{tol}: {acc:.4f}")

        self.test_preds_np = preds
        self.test_targets_np = targets
        self.test_metrics = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "accuracy_tol": acc,
        }

        if self.hparams.save_predictions:
            import pandas as pd
            df = pd.DataFrame({"target": targets, "prediction": preds})
            df.to_csv(self.hparams.predictions_path, index=False)
            print(f"\nPredictions saved to {self.hparams.predictions_path}")

    # -------- optimizers --------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True,
        )

        # still monitor val_mse (or val_loss if you prefer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_mse",
            },
        }
