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
                 image_path: str,
                 mask_path: str,
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
        self.image_pathes_path = Path(self.path_to_save) / str(self.patch_size)/ 'images'
        self.mask_pathes_path = Path(self.path_to_save) / str(self.patch_size) / 'masks'
        
        if self.image_pathes_path.exists() and self.mask_pathes_path.exists():
            self.images = np.array([cv2.cvtColor(cv2.imread(str(img)), cv2.COLOR_BGR2RGB) for img in self.image_pathes_path.glob("*.jpg")])
            self.masks = np.array([cv2.imread(str(img), 0) for img in self.mask_pathes_path.glob("*.jpg")])
        else:
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

    def save_patches(self):
        if self.patch_size is None:
            raise ValueError('You must specify patch size.')
        tqd = tqdm(enumerate(zip(self.images, self.masks), start=1))

        for i, (image, mask) in tqd:
            self.image_pathes_path.mkdir(exist_ok=True, parents=True)
            self.mask_pathes_path.mkdir(exist_ok=True, parents=True)

            cv2.imwrite(str(self.image_pathes_path / f'{i}.jpg'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(self.mask_pathes_path / f'{i}.jpg'), mask)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = np.expand_dims(self.masks[idx], -1)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image'].float()
            mask = transformed['mask'].float().div(255).round()

        return image, mask
