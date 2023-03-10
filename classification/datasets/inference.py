from torch.utils.data import Dataset, DataLoader
import cv2
import os
import torch
import pandas as pd
from datasets.transforms import preprocess


class TorchDataset(Dataset):
    def __init__(
            self,
            csv_file,
            img_dir,
            config,
    ):
        self.img_dir = os.path.join(config['data_path'], img_dir)
        self.config = config
        self.__read_csv(os.path.join(config['data_path'], csv_file))
        self.__init_transforms(**config['preprocess'])

    def __init_transforms(
            self, side_size
    ):
        self.preprocess = preprocess(side_size)

    def __read_csv(self, csv_file):
        self.images = pd.read_csv(csv_file).filename.tolist()

    def __len__(self):
        return len(self.images)

    def load_sample(self, idx):
        image_path = self.images[idx]
        image = cv2.imread(os.path.join(self.img_dir, image_path))
        return image

    def __getitem__(self, idx):
        image = self.load_sample(idx)
        image = self.preprocess(image=image)['image']
        return image, idx


def create_inference_loader_dataset(
        csv_file,
        img_dir,
        config
):
    dataset = TorchDataset(
        csv_file,
        img_dir,
        config
    )

    dataloader = DataLoader(
        dataset, collate_fn=collate_fn,
        **config['dataloader']
    )
    return dataloader, dataset


def collate_fn(batch):
    inputs = torch.stack([sample[0] for sample in batch])
    ids = torch.as_tensor([sample[1] for sample in batch])
    return inputs, ids
