import argparse
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.plot import reshape_as_image


def read_raster(raster_path: str):
    with rasterio.open(raster_path, 'r', driver='JP2OpenJPEG') as src:
        raster_image = src.read()
    return raster_image


def make_patches(image: bytes,
                 patch_size: int,
                 width: int,
                 height: int,
                 save_path: str,
                 mask: bool = False):
    save_path = Path(save_path)

    def get_save_str(height_cord: int, width_cord: int, pixel_ratio: float):
        parent_path = save_path / str(patch_size)
        parent_path.mkdir(exist_ok=True, parents=True)
        return str(parent_path / f'x_{height_cord}y_{width_cord}_{pixel_ratio}.jpg')

    def get_pixel_ratio(clip_image):
        rate = clip_image.sum() / patch_size ** 2
        return 1 if rate > 0 else 0

    for h_cord in np.arange(start=patch_size,
                            stop=height + 1,
                            step=patch_size):
        for w_cord in np.arange(start=patch_size,
                                stop=width + 1,
                                step=patch_size):

            clipped_img = reshape_as_image(image[:, h_cord - patch_size: h_cord, w_cord - patch_size: w_cord])
            pixel_ratio = 1
            if mask:
                pixel_ratio = get_pixel_ratio(clipped_img)

            if not mask:
                clipped_img = cv2.cvtColor(clipped_img, cv2.COLOR_RGB2BGR)

            if cv2.imwrite(get_save_str(h_cord, w_cord, pixel_ratio), clipped_img) is False:
                raise cv2.error


def main(args):
    image = read_raster(args.image_path)
    mask = read_raster(args.mask_path)

    i_dim, i_width, i_height = image.shape
    m_dim, m_width, m_height = mask.shape

    assert i_dim == 3, "Image channels not equal to 3"
    assert m_dim == 1, "Mask channels not equal to 1"
    assert i_width == m_width and i_height == m_height, "Image and mask have different shapes"

    try:
        make_patches(image, args.patch_size, i_width, i_height, args.images_dir, False)
        make_patches(mask, args.patch_size, i_width, i_height, args.masks_dir, True)
        print('Images saved successfully')
    except cv2.error as e:
        print('Error saving image')


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(add_help=add_help)
    parser.add_argument("--image_path", default="data/raw/image.jp2", type=str, help="image raster path")
    parser.add_argument("--mask_path", default="data/raw/mask.jp2", type=str, help="mask raster path")
    parser.add_argument("--patch_size", default=256, type=int, help="desired patch size")
    parser.add_argument("--images_dir", default="data/images", type=str, help="path to save new images")
    parser.add_argument("--masks_dir", default="data/masks", type=str, help="path to save new masks")
    return parser


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
