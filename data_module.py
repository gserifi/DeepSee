from pathlib import Path
from typing import List, Tuple

import lightning as lit
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ImageAndDepthDataset(Dataset):
    def __init__(self, image_dir: Path, image_list: List[Tuple[str, str]]):
        self.image_dir = image_dir
        self.image_list = image_list

    def __len__(self) -> int:
        return len(self.image_list)

    @staticmethod
    def load_image(path: Path) -> torch.Tensor:
        im = Image.open(path).convert("RGB")
        im = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0

        return im

    @staticmethod
    def load_depth(path: Path) -> torch.Tensor:
        if not path.exists():
            return torch.zeros(1, 426, 560)
        dp = np.load(path)
        dp = torch.from_numpy(dp).unsqueeze(-1).permute(2, 0, 1)

        return dp

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, depth_path = self.image_list[idx]
        image = ImageAndDepthDataset.load_image(self.image_dir / image_path)
        depth = ImageAndDepthDataset.load_depth(self.image_dir / depth_path)

        return image, depth


class LitDataModule(lit.LightningDataModule):
    train_list: List[Tuple[str, str]]
    val_list: List[Tuple[str, str]]
    test_list: List[Tuple[str, str]]

    train_dataset: ImageAndDepthDataset
    val_dataset: ImageAndDepthDataset
    test_dataset: ImageAndDepthDataset

    def __init__(
        self,
        data_root: Path = Path("./data"),
        batch_size=32,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    ):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory

    def prepare_data(self):
        with open(self.data_root / "train_list.txt", "r") as f:
            self.train_list = [
                (paths[0].strip(), paths[1].strip())
                for line in f.readlines()
                if (paths := line.split(" "))
            ]

        split = int(len(self.train_list) * 0.8)
        self.val_list = self.train_list[split:]
        self.train_list = self.train_list[:split]

        with open(self.data_root / "test_list.txt", "r") as f:
            self.test_list = [
                (paths[0].strip(), paths[1].strip())
                for line in f.readlines()
                if (paths := line.split(" "))
            ]

        print(
            f"#Train: {len(self.train_list)}, #Val: {len(self.val_list)}, #Test: {len(self.test_list)}"
        )

    def setup(self, stage=None):
        self.train_dataset = ImageAndDepthDataset(
            self.data_root / "train" / "train", self.train_list
        )
        self.val_dataset = ImageAndDepthDataset(
            self.data_root / "train" / "train", self.val_list
        )
        self.test_dataset = ImageAndDepthDataset(
            self.data_root / "test" / "test", self.test_list
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )
