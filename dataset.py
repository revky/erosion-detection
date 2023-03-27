from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import rasterio
from rasterio.plot import reshape_as_image
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class SegmentDataset(Dataset):
    def __init__(self,
                 image_path: Path,
                 mask_path: Path,
                 patch_size: int,
                 transform: A.Compose = None,
                 save_images=False,
                 path_to_save=None):
        super().__init__()
        self.image_path = image_path
        self.mask_path = mask_path
        self.patch_size = patch_size
        self.transform = transform
        self.save_images = save_images
        self.path_to_save = path_to_save

        with rasterio.open(self.image_path, 'r', driver='JP2OpenJPEG') as src:
            self.image_raster = src.read()

        with rasterio.open(self.mask_path, 'r', driver='JP2OpenJPEG') as src:
            self.mask_raster = src.read()

        if self.image_raster[0].shape == self.mask_raster[0].shape:
            self.width = self.image_raster.shape[1]
            self.height = self.image_raster.shape[2]

            self.images = self.make_patches(self.image_raster)
            self.masks = self.make_patches(self.mask_raster)

        if self.save_images:
            self.save_patches()

    def make_patches(self, raster: np.array):
        slices = []
        for h_cord in np.arange(start=self.patch_size,
                                stop=self.height + 1,
                                step=self.patch_size):
            for w_cord in np.arange(start=self.patch_size,
                                    stop=self.width + 1,
                                    step=self.patch_size):
                slices.append(
                    reshape_as_image(raster[:, h_cord - self.patch_size: h_cord, w_cord - self.patch_size: w_cord]))
        return slices

    # TODO
    # If images found load from cashe
    def save_patches(self):
        if self.patch_size is None:
            raise ValueError('You must specify patch size.')
        tqd = tqdm(enumerate(zip(self.images, self.masks), start=1))

        for i, (image, mask) in tqd:
            parent = Path(self.path_to_save) / str(self.patch_size)
            img_path = parent / 'images'
            mask_path = parent / 'masks'

            img_path.mkdir(exist_ok=True, parents=True)
            mask_path.mkdir(exist_ok=True, parents=True)

            cv2.imwrite(str(img_path / f'{i}.jpg'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(mask_path / f'{i}.jpg'), mask)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return image, mask.float().div(255).round()
