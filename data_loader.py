import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import List


class DataLoaderClass(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 4,
        pin_memory: bool = False,
        drop_last: bool = False,
        config: dict = None,
    ) -> None:
        super(DataLoaderClass, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        self.config = config or {}

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> "IteratorClass":
        return IteratorClass(self)


class IteratorClass(object):
    def __init__(self, data_loader: DataLoaderClass) -> None:
        self.dl = data_loader
        self.dataset = data_loader.dataset
        self.batch_size = data_loader.batch_size
        self.shuffle = data_loader.shuffle
        self.num_workers = data_loader.num_workers
        self.pin_memory = data_loader.pin_memory
        self.index = 0
        self.dataset_iter = iter(self.dataset)

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        try:
            if self.shuffle:
                self.dataset.shuffle()
            batch = next(self.dataset_iter)
            if self.pin_memory:
                batch = [t.pin_memory() for t in batch]
            return batch
        except StopIteration:
            self.index += 1
            if self.drop_last and self.index > len(self.dl):
                raise StopIteration
            return self.__next__()


class DatasetClass(Dataset):
    def __init__(self, data_dir: str, transforms_: List[transforms.Transform] = None) -> None:
        self.data_dir = data_dir
        self.transforms = transforms_ or []

        self.images = []
        self.labels = []

        self.load_data()

    def load_data(self) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> List[torch.Tensor]:
        image_path = self.images[index]
        image = Image.open(image_path)

        for t in self.transforms:
            image = t(image)

        image = np.array(image)
        image = torch.from_numpy(image)

        label = torch.tensor(self.labels[index])

        return [image, label]


class ImageFolderDataset(DatasetClass):
    def load_data(self) -> None:
        image_paths = [
            os.path.join(self.data_dir, x)
            for x in sorted(os.listdir(self.data_dir))
            if os.path.isfile(os.path.join(self.data_dir, x))
        ]

        self.images = image_paths
        self.labels = [int(x[2]) for x in image_paths]


class CSVDataset(DatasetClass):
    def load_data(self) -> None:
        df = pd.read_csv(self.data_dir)

        self.images = df["image_path"].tolist()
        self.labels = df["label"].tolist()


def create_data_loader(
    data_dir: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    drop_last: bool,
    config: dict = None,
) -> DataLoaderClass:
    transform_chain = get_transformations()

    dataset = ImageFolderDataset(data_dir, transform_chain)

    return DataLoaderClass(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        config=config,
    )


def get_transformations() -> List[transforms.Transform]:
    transform_list = []
    transform_list.append(transforms.ToPILImage())
    transform_list.append(transforms.RandomRotation(10))
    transform_list.append(transforms.RandomResizedCrop(224))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transform_list