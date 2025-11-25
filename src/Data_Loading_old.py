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
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
from torchvision.transforms import v2
import pandas as pd

from src.index_dataset import scan_dataset

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


def getLRDDDataLoader(data_root, metadata_dir, batch_size, num_workers, shuffle=True):

    manifest = scan_dataset(data_root, metadata_dir)
    manifest_df = pd.DataFrame(manifest)
    
    list_of_datasets = []
    for _, row in manifest_df.iterrows():
        try:
            list_of_datasets.append(LRDDDatasetChunk(row["csv_path"], row["img_dir"], row["label_dir"], transform=None))       
        except Exception as e:
            print(f"[extract:ERROR] {row['date']}/{row['flight_id']} failed: {e}")

    full_dataset = data.ConcatDataset(list_of_datasets)
    loader = DataLoader(dataset=full_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    return loader


class LRDDDatasetChunk(Dataset):
    """
    Dataset for a single flight.

    Inputs:
        csv_file: path to metadata CSV (must include columns like img_name, distance_3d_ft)
        img_folder: directory with the RGB frames
        labels_folder: directory with YOLO txt labels
        transform: (optional) torchvision v2 transform

    __getitem__ returns:
        full_img_tensor: (3,224,224) float32 normalized
        crop_tensor:     (3,224,224) float32 normalized (drone crop if we have bbox, else full img)
        bbox_features:   tensor([width, height, area, aspect]) in normalized coords
        distance:        scalar tensor (float32), distance_3d_ft
    """

    def __init__(self, csv_file, img_folder, labels_folder, transform=None):
        
        # assign first so they're available in checks
        self.imgs = img_folder
        self.yolo_labels = labels_folder

        df = pd.read_csv(csv_file)

        keep_mask = []
        missing_image_count = 0
        missing_distance_count = 0

        for _, row in df.iterrows():
            img_name = row["img_name"]
            img_path = os.path.join(self.imgs, img_name)

            # 1. check image exists
            if not os.path.exists(img_path):
                missing_image_count += 1
                keep_mask.append(False)
                continue

            # 2. check distance is valid numeric
            raw_dist = row.get("distance_3d_ft", None)

            # helper: is this usable distance?
            def valid_distance(val):
                # reject None / NaN
                if val is None or pd.isna(val):
                    return False
                # try to convert to float
                try:
                    float_val = float(val)
                except (TypeError, ValueError):
                    return False
                # optional sanity: distance should be >0
                # tweak if you ever have 0-ft ground truth
                if not np.isfinite(float_val):
                    return False
                return True

            if not valid_distance(raw_dist):
                missing_distance_count += 1
                keep_mask.append(False)
                continue

            # if we got here, keep the row
            keep_mask.append(True)

        df_clean = df[keep_mask].reset_index(drop=True)
        self.metadata = df_clean

        dropped_total = missing_image_count + missing_distance_count
        if dropped_total > 0:
            if missing_image_count > 0:
                print(
                    f"[LRDDDataset] WARN: {missing_image_count} rows dropped "
                    f"because image file was missing in {self.imgs}"
                )
            if missing_distance_count > 0:
                print(
                    f"[LRDDDataset] WARN: {missing_distance_count} rows dropped "
                    f"because distance_3d_ft was missing/invalid in {os.path.basename(csv_file)}"
                )
            print(
                f"[LRDDDataset] INFO: kept {len(df_clean)} / {len(df)} rows "
                f"for {os.path.basename(csv_file)}"
            )

        self.metadata = df_clean
        self.yolo_labels = labels_folder

        if transform is None:
            transform = v2.Compose([
                v2.Resize((720,720)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ) # imagenet mean and std
            ])
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_name = row['img_name']

        # --- open image ---
        img_path = os.path.join(self.imgs, img_name)
        img = Image.open(img_path).convert('RGB')
        img_width, img_height = img.size

        # --- try to read YOLO bbox from txt ---
        label_path = os.path.join(
            self.yolo_labels,
            os.path.splitext(img_name)[0] + ".txt"
        )

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                line = f.readline().strip()

            if not line:
                crop = img
                width = height = area = aspect = 0.0
            else:
                parts = line.split()
                # YOLO format: cls cx cy w h (all normalized 0-1)
                _, x_center, y_center, width, height = map(float, parts)

                # convert to pixels
                x_center_px = x_center * img_width
                y_center_px = y_center * img_height
                w_px = width  * img_width
                h_px = height * img_height

                x_min = x_center_px - w_px/2
                y_min = y_center_px - h_px/2
                x_max = x_center_px + w_px/2
                y_max = y_center_px + h_px/2

                crop = img.crop((x_min, y_min, x_max, y_max))

                area = width * height
                aspect = width / (height + 1e-6)
        else:
            crop = img
            width = height = area = aspect = 0.0

        full_img_tensor = self.transform(img)
        crop_tensor     = self.transform(crop)

        distance = torch.tensor(row['distance_3d_ft'], dtype=torch.float32)
        bbox_features = torch.tensor([width, height, area, aspect], dtype=torch.float32)

        return full_img_tensor, crop_tensor, bbox_features, distance