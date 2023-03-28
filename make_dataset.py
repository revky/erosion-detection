import rasterio
from pathlib import Path

def read_raster(raster_path:str):
    with rasterio.open(raster_path, 'r', driver='JP2OpenJPEG') as src:
        raster_image = src.read()
        raster_meta = src.meta()
    return raster_image, raster_meta
