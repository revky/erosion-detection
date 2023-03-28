from typing import Union
from pathlib import Path
import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class SegmentDataset(Dataset):
    def __init__(self,
                 images_dir: Path,
                 masks_dir: Path,
                 transform: Union[A.Compose, ToTensorV2] = None,
                 empty_fraction: int = None):
        super().__init__()
        self.images = sorted(images_dir.glob("*.jpg"))
        self.masks = sorted(masks_dir.glob("*.jpg"))
        self.transform = transform
        self.empty_fraction = empty_fraction

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(str(self.images[idx])), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.masks[idx]), cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image'].float().div(255)
            mask = transformed['mask'].div(255).round()
        return image, mask
