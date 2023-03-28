from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.plot import reshape_as_image


def read_raster(raster_path: str):
    with rasterio.open(raster_path, 'r', driver='JP2OpenJPEG') as src:
        raster_image = src.read()
    return raster_image


def make_patches(image: bytes, patch_size: int, save_path: str, mask=False):
    _, width, height = image.shape
    save_path = Path(save_path)

    def get_save_str(height_cord: int, width_cord: int):
        return str(save_path / str(patch_size) / f'x_{height_cord}y_{width_cord}.jpg')

    for h_cord in np.arange(start=patch_size,
                            stop=height + 1,
                            step=patch_size):
        for w_cord in np.arange(start=patch_size,
                                stop=width + 1,
                                step=patch_size):
            clipped_img = reshape_as_image(image[:, h_cord - patch_size: h_cord, w_cord - patch_size: w_cord])
            if not mask:
                clipped_img = cv2.cvtColor(clipped_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                get_save_str(h_cord, w_cord),
                clipped_img
            )
