import re
from pathlib import Path
from typing import Union

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
                 add_empty_masks: int = None):
        super().__init__()
        self.images = np.array(sorted(images_dir.glob("*.jpg")))
        self.masks = np.array(sorted(masks_dir.glob("*.jpg")))
        self.transform = transform
        self.add_empty_masks = add_empty_masks

        if self.add_empty_masks:
            binary_mask = np.ma.make_mask([int(re.compile(r'\d\.').search(str(mask))[0][:-1]) for mask in self.masks])
            false_indices = np.where(binary_mask is False)[0]
            if self.add_empty_masks > len(false_indices):
                self.add_empty_masks = len(false_indices)
            indices_to_invert = np.random.choice(false_indices, size=self.add_empty_masks, replace=False)

            binary_mask[indices_to_invert] = True
            self.images = self.images[binary_mask]
            self.masks = self.masks[binary_mask]

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
