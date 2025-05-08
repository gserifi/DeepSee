from pathlib import Path
from typing import Optional, Union

import lightning as lit
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ImageAndDepthDatasetItem = tuple[torch.Tensor, Union[torch.Tensor, str]]


class ImageAndDepthDataset(Dataset[ImageAndDepthDatasetItem]):
    """
    Dataset for loading images and depth maps. __getitem__ returns a tuple of (image, depth), where image is always a
    tensor and depth may be either a tensor or a string denoting the path to the depth file (if ground truth is not
    available, e.g. for the prediction set).
    """

    def __init__(self, image_dir: Path, image_list: list[tuple[str, str]]):
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
    def load_depth(path: Path) -> Optional[torch.Tensor]:
        if not path.exists():
            return None

        dp = np.load(path)
        dp = torch.from_numpy(dp).unsqueeze(-1).permute(2, 0, 1)

        return dp

    def __getitem__(self, idx) -> ImageAndDepthDatasetItem:
        image_path, depth_path = self.image_list[idx]
        image = ImageAndDepthDataset.load_image(self.image_dir / image_path)
        depth = ImageAndDepthDataset.load_depth(self.image_dir / depth_path)

        if depth is None:
            depth = depth_path

        return image, depth


class LitDataModule(lit.LightningDataModule):
    """
    Data module for loading image and depth datasets. See `ImageAndDepthDataset` for details.
    """

    train_list: list[tuple[str, str]]
    val_list: list[tuple[str, str]]
    test_list: list[tuple[str, str]]
    predict_list: list[tuple[str, str]]

    train_dataset: ImageAndDepthDataset
    val_dataset: ImageAndDepthDataset
    test_dataset: ImageAndDepthDataset
    predict_dataset: ImageAndDepthDataset

    def __init__(
        self,
        data_root: Path = Path("./data"),
        batch_size: int = 32,
        num_workers: int = 4,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        train_split: float = 0.7,  # 70% train, 30% val + split
        holdout_split: float = 0.33,  # 33% val, 67% test
    ):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.train_split = train_split
        self.holdout_split = holdout_split

    def prepare_data(self):
        with open(self.data_root / "train_list.txt", "r") as f:
            self.train_list = [
                (paths[0].strip(), paths[1].strip())
                for line in f.readlines()
                if (paths := line.split(" "))
            ]

        split = int(len(self.train_list) * self.train_split)
        holdout_list = self.train_list[split:]
        self.train_list = self.train_list[:split]

        split = int(len(holdout_list) * self.holdout_split)
        self.val_list = holdout_list[:split]
        self.test_list = holdout_list[split:]

        with open(self.data_root / "test_list.txt", "r") as f:
            self.predict_list = [
                (paths[0].strip(), paths[1].strip())
                for line in f.readlines()
                if (paths := line.split(" "))
            ]

        print(
            f"#Train: {len(self.train_list)}, #Val: {len(self.val_list)}, #Test: {len(self.test_list)}, #Predict: {len(self.predict_list)}"
        )

    def setup(self, stage=None):
        self.train_dataset = ImageAndDepthDataset(
            self.data_root / "train" / "train", self.train_list
        )
        self.val_dataset = ImageAndDepthDataset(
            self.data_root / "train" / "train", self.val_list
        )
        self.test_dataset = ImageAndDepthDataset(
            self.data_root / "train" / "train", self.test_list
        )
        self.predict_dataset = ImageAndDepthDataset(
            self.data_root / "test" / "test", self.predict_list
        )

    def dataloader(
        self, dataset: ImageAndDepthDataset, shuffle: bool = False
    ) -> DataLoader[ImageAndDepthDatasetItem]:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self) -> DataLoader[ImageAndDepthDatasetItem]:
        return self.dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader[ImageAndDepthDatasetItem]:
        return self.dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader[ImageAndDepthDatasetItem]:
        return self.dataloader(self.test_dataset, shuffle=False)

    def predict_dataloader(self) -> DataLoader[ImageAndDepthDatasetItem]:
        return self.dataloader(self.predict_dataset, shuffle=False)
