import os
import sys
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt


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
