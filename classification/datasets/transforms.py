import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def transforms(probs=0.25):
    return A.Compose([
        A.RandomBrightnessContrast(p=probs, brightness_by_max=True),
    ])


def preprocess(side_size=256):
    return A.Compose([
        A.PadIfNeeded(
            min_height=side_size,
            min_width=side_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        ),
        A.Normalize(),
        ToTensorV2()
    ])
