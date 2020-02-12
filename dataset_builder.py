import matplotlib.pyplot as plt
from pysheds.grid import Grid
import pyproj
import numpy as np
import cv2
from skimage.util import view_as_blocks
from pathlib import Path
import georasters as gr
from geopandas.io import file
import cv2
from skimage.morphology import skeletonize
#


def extract_patches_from_raster():
    count = 0
    for raster_file in Path('./world_map').glob('**/*.tif'):
        data = gr.from_file(str(raster_file))
        raster_blocks = view_as_blocks(data.raster, (225, 225))
        for i in range(raster_blocks.shape[0]):
            for j in range(raster_blocks.shape[1]):
                raster_data = raster_blocks[i, j]

                src = cv2.pyrDown(
                    raster_data,
                    dstsize=(
                        raster_data.shape[1] // 2,
                        raster_data.shape[0] // 2))

                data_out_downsampled = gr.GeoRaster(
                    src,
                    data.geot,
                    nodata_value=data.nodata_value,
                    projection=data.projection,
                    datatype=data.datatype,
                )
                data_out_downsampled.to_tiff(
                    './data_downsampled_blurred/data_q' + str(count) + str(i) + str(j))

                data_out = gr.GeoRaster(
                    raster_data,
                    data.geot,
                    nodata_value=data.nodata_value,
                    projection=data.projection,
                    datatype=data.datatype,
                )
                data_out.to_tiff(
                    './data/data_q' + str(count) + str(i) + str(j))
                count += 1


def compute_rivers(tiff_image):
    grid = Grid.from_raster(str(tiff_image), data_name='dem')
    depressions = grid.detect_depressions('dem')

    grid.fill_depressions(data='dem', out_name='flooded_dem')
    flats = grid.detect_flats('flooded_dem')
    grid.resolve_flats(data='flooded_dem', out_name='inflated_dem')

    # Compute flow direction based on corrected DEM
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    grid.flowdir(data='inflated_dem', out_name='dir', dirmap=dirmap)
    # Compute flow accumulation based on computed flow direction
    grid.accumulation(data='dir', out_name='acc', dirmap=dirmap)
    downsampled_rivers = np.log(grid.view('acc') + 1)
    upsampled_depressions = cv2.pyrUp(
        np.array(depressions, dtype=np.uint8),
        dstsize=(225, 225))

    upsampled_rivers = cv2.pyrUp(
        downsampled_rivers,
        dstsize=(225, 225))
    upsampled_rivers = (upsampled_rivers - np.amin(upsampled_rivers)) / \
        (np.amax(upsampled_rivers) - np.amin(upsampled_rivers))
    upsampled_rivers = np.array(upsampled_rivers * 255, dtype=np.uint8)
    _, thresholded_river = cv2.threshold(upsampled_rivers, 127, 255, cv2.THRESH_BINARY)
    thresholded_river[thresholded_river == 255] = 1
    skeletonized_rivers = skeletonize(thresholded_river)

    return np.expand_dims(skeletonized_rivers, axis=-
                          1), np.expand_dims(upsampled_depressions, axis=-1)


def compute_ridges(tiff_image):

    grid = Grid.from_raster(str(tiff_image), data_name='dem')
    grid.dem = grid.dem.max() - grid.dem
    peaks = grid.detect_depressions('dem')
    grid.fill_depressions(data='dem', out_name='flooded_dem')
    flats = grid.detect_flats('flooded_dem')
    grid.resolve_flats(data='flooded_dem', out_name='inflated_dem')

    # Compute flow direction based on corrected DEM
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    grid.flowdir(data='inflated_dem', out_name='dir', dirmap=dirmap)
    # Compute flow accumulation based on computed flow direction
    grid.accumulation(data='dir', out_name='acc', dirmap=dirmap)
    downsampled_ridges = np.log(grid.view('acc') + 1)
    upsampled_peaks = cv2.pyrUp(
        np.array(peaks, dtype=np.uint8),
        dstsize=(225, 225))
    upsampled_ridges = cv2.pyrUp(
        downsampled_ridges,
        dstsize=(225, 225))
    upsampled_ridges = (upsampled_ridges - np.amin(upsampled_ridges)) / \
        (np.amax(upsampled_ridges) - np.amin(upsampled_ridges))
    upsampled_ridges = np.array(upsampled_ridges * 255, dtype=np.uint8)
    _, thresholded_ridges = cv2.threshold(upsampled_ridges, 150, 255, cv2.THRESH_BINARY)
    thresholded_ridges[thresholded_ridges == 255] = 1
    skeletonized_ridges = skeletonize(thresholded_ridges)

    return np.expand_dims(skeletonized_ridges, axis=-1), np.expand_dims(upsampled_peaks, axis=-1)


def compute_sketches():
    count = 0
    height_maps = []
    sketch_maps = []
    for filename in Path('./data_downsampled_blurred').glob('**/*.tif'):
        file_path = str(filename)
        file_id = file_path.split('/')
        detailed_data = gr.from_file('./data/' + file_id[-1])
        data = gr.from_file(str(filename))
        if data.mean() < 5:
            continue
        ridges, peaks = compute_ridges(filename)
        rivers, basins = compute_rivers(filename)
        height_map = np.array(detailed_data.raster, dtype=np.float32)
        height_map = np.expand_dims(height_map, axis=-1)
        height_map = (height_map - np.amin(height_map)) / \
            (np.amax(height_map) - np.amin(height_map))
        height_map = height_map * 2 - 1

        sketch_map = np.stack((ridges, rivers, peaks, basins), axis=2)
        sketch_map = np.squeeze(sketch_map, axis=-1)
        print(sketch_map.shape)
        height_maps.append(height_map)
        sketch_maps.append(sketch_map)
    training_output = np.array(height_maps, dtype=np.float32)
    training_input = np.array(sketch_maps, dtype=np.float32)
    np.savez('training_data.npz', x=training_input, y=training_output)
