# src/Data_Loading.py

import os
import pytorch_lightning as pl

from src.dataset import getLRDDDataLoader


class DroneDataModule(pl.LightningDataModule):
    """
    Lightning DataModule that wraps your original getLRDDDataLoader.

    It uses the same folder layout as your previous code:
        ROOT_IMAGES   = "/mnt/archive/LRDDv3"
        ROOT_METADATA = "/mnt/archive/LRDDv3/metadata"

    and expects subfolders:
        train/, val/, test/ under both images and metadata roots.
    """

    def __init__(
        self,
        root_images: str,
        root_metadata: str,
        batch_size: int = 16,
        num_workers: int = 4,
        img_width=1280,
        img_height=1280
    ):
        super().__init__()
        self.root_images = root_images
        self.root_metadata = root_metadata
        self.batch_size = batch_size
        self.num_workers = num_workers

        # We'll store the loaders so we only build them once in setup()
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None
        
        self.img_width = img_width
        self.img_height = img_height

    def setup(self, stage=None):
        """
        Make dataloaders for train/val/test using the exact same
        getLRDDDataLoader logic as before.
        """
        if stage == "fit" or stage is None:
            train_images = os.path.join(self.root_images, "test")
            train_meta = os.path.join(self.root_metadata, "test")

            val_images = os.path.join(self.root_images, "val")
            val_meta = os.path.join(self.root_metadata, "val")

            self._train_loader = getLRDDDataLoader(
                train_images,
                train_meta,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                img_width=self.img_width,
                img_height=self.img_height
            )

            self._val_loader = getLRDDDataLoader(
                val_images,
                val_meta,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                img_width=self.img_width,
                img_height=self.img_height
            )

        if stage == "test" or stage is None:
            test_images = os.path.join(self.root_images, "test")
            test_meta = os.path.join(self.root_metadata, "test")

            self._test_loader = getLRDDDataLoader(
                test_images,
                test_meta,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                img_width=self.img_width,
                img_height=self.img_height
            )

    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
        return self._val_loader

    def test_dataloader(self):
        return self._test_loader
