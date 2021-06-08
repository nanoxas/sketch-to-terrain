import itertools
import os
from pathlib import Path

import cv2
import georasters as gr
import numpy as np
from PIL import Image
from pysheds.grid import Grid
from skimage.morphology import skeletonize
from skimage.util import view_as_blocks
from tqdm import tqdm


def extract_patch_simple_map():
    count = 0
    base_save_path = Path('./data_satelite/')
    # TODO: move filtering information at the beginning of data processing (add metadata to trainin data)
    for map_part in Path('./world_map_september').glob('**/*.png'):
        print('Processing: ' + str(map_part))
        data = np.asarray(Image.open(str(map_part)))
        raster_blocks = view_as_blocks(data, (225, 225, 3))
        t_total = np.product(raster_blocks.shape[:2])
        for i, j in tqdm(itertools.product(range(raster_blocks.shape[0]), range(raster_blocks.shape[1])),
                         total=t_total):
            block_id = str(count) + str(i) + str(j)
            data_reference = gr.from_file('./data_downsampled_blurred/data_q' + block_id + '.tif')
            if data_reference.mean() < 5:
                count += 1
                continue
            Image.fromarray(raster_blocks[i, j].squeeze()).save(str(base_save_path.joinpath(block_id)) + '.png')
            count += 1


def recombine_heightmap_sattelite_data():
    # TODO: make sure the ordering is the same as in heightmap
    satelite_data = []
    for idx, filename in tqdm(enumerate(Path('./data_satelite').glob('**/*.png'))):
        satelite_data.append(np.asarray(Image.open(filename)))
    satelite_datta_arr = np.array(satelite_data)
    data = np.load('training_data.npz')
    np.savez('training_data.npz',
             sketches=data['sketches'],
             heightmaps=data['heightmaps'],
             satellites=satelite_datta_arr)


def extract_patches_from_raster():
    count = 0
    for raster_file in Path('./world_map').glob('**/*.TIF'):
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
                    fill_value=-10,
                    nodata_value=-1,
                    projection=data.projection,
                    datatype=data.datatype,
                )
                data_out_downsampled.to_tiff(
                    './data_downsampled_blurred/data_q' + str(count) + str(i) + str(j))

                data_out = gr.GeoRaster(
                    raster_data,
                    data.geot,
                    fill_value=-10,
                    nodata_value=-1,
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
    height_maps = []
    sketch_maps = []
    for filename in Path('./data_downsampled_blurred').glob('**/*.tif'):
        file_path = str(filename)
        file_id = file_path.split(os.sep)[-1]
        detailed_data = gr.from_file(os.path.join('./data/', file_id))
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
    height_maps = np.array(height_maps, dtype=np.float32)
    sketch_maps = np.array(sketch_maps, dtype=np.float32)
    np.savez('training_data.npz', sketches=sketch_maps, heightmaps=height_maps)


if __name__ == "__main__":
    extract_patches_from_raster()
    extract_patch_simple_map()
    compute_sketches()
    recombine_heightmap_sattelite_data()
