import json
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import torch
from datasets.transforms import preprocess, transforms


class TorchDataset(Dataset):
    def __init__(
            self,
            csv_file,
            img_dir,
            config,
            mode='train'
    ):
        self.images = []
        self.img_dir = os.path.join(config["data_path"], img_dir)
        self.mode = mode
        self.config = config
        self.__read_csv(os.path.join(config["data_path"], csv_file))
        self.__init_transforms(**config['transforms'])

    def __read_csv(self, csv_file):
        with open(csv_file, 'r') as file:
            for i, line in enumerate(file.readlines()):
                if i == 0:
                    continue
                image, label = line.strip().split(',')
                self.images.append([image, int(label)])

    def __init_transforms(
            self, aug_prob, side_size
    ):
        self.preprocess = preprocess(side_size)
        self.transforms = transforms(aug_prob) if self.mode == "train" else None

    def __len__(self):
        return len(self.images)

    def load_sample(self, idx):
        image_path, label = self.images[idx]
        if not os.path.exists(os.path.join(self.img_dir, image_path)):
            raise ValueError(f"{os.path.join(self.img_dir, image_path)} doesn't exist")
        image = cv2.imread(os.path.join(self.img_dir, image_path))
        return image, label

    def __getitem__(self, idx):
        image, label = self.load_sample(idx)
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        image = self.preprocess(image=image)['image']
        return image, torch.as_tensor(label)


def collate_fn(batch):
    items = list(zip(*batch))
    return [torch.stack(item) for item in items]


def create_loader_dataset(
        csv_file,
        img_dir,
        config,
        mode='train'
):
    dataset = TorchDataset(
        csv_file, img_dir, config, mode=mode
    )

    dataloader = DataLoader(
        dataset,
        shuffle=(mode == "train"),
        collate_fn=collate_fn,
        **config['dataloader']
    )
    return dataloader, dataset
