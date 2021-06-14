import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from model import TerrainGANBuilder
import argparse

PATCH_SIZE = 15


def generate_fake_samples(generator, dataset, patch_size):
    w_noise = np.random.normal(0, 1, (dataset.shape[0], 14, 14, 1024))
    x = generator.predict([dataset, w_noise])
    y = np.zeros((len(x), patch_size, patch_size, 1))
    return x, y


def generate_real_samples(x, y, n_samples, patch_size):
    ix = np.random.randint(0, x.shape[0], n_samples)
    x_real = x[ix]
    y_real = y[ix]
    labels = np.ones((n_samples, patch_size, patch_size, 1))
    return x_real, y_real, labels


def delegate_mode_building(builder_obj: TerrainGANBuilder, optimizer, mode: str):
    if mode == 'SketchToHeightmap':
        return builder_obj.build_sketch_to_terrain(optimizer)
    elif mode == 'SketchToSatellite':
        return builder_obj.build_sketch_to_satelite(optimizer, sequential=False)
    elif mode == 'HeightmapToSatellite':
        return builder_obj.build_terrain_to_satelite(optimizer, sketches=False)
    elif mode == 'SketchHeightmapToSatellite':
        return builder_obj.build_terrain_to_satelite(optimizer, sketches=True)
    else:
        raise ValueError(f'Unrecognised mode: {mode}')


def compile_training_data(sketches: np.ndarray, heightmaps, satellites, mode: str):
    if mode == 'SketchToHeightmap':
        return sketches, heightmaps
    elif mode == 'SketchToSatellite':
        return sketches, np.concatenate([heightmaps, satellites / 255.0], axis=-1)
    elif mode == 'HeightmapToSatellite':
        return heightmaps, satellites / 255.0
    elif mode == 'SketchHeightmapToSatellite':
        return np.concatenate([heightmaps, sketches], axis=-1), satellites / 255.0
    else:
        raise ValueError(f'Unrecognised mode: {mode}')


def transform_back_input_img(img: np.ndarray, mode: str) -> np.ndarray:
    if mode in ['SketchToHeightmap', 'SketchToSatellite']:
        return img
    elif mode == 'HeightmapToSatellite':
        return np.uint8(img * 127.5 + 127.5)
    elif mode == 'SketchHeightmapToSatellite':
        heightmap = np.expand_dims(np.uint8(img[..., 0] * 127.5 + 127.5), axis=-1)
        satellite = np.uint8(img[..., 1:] * 255.0)
        return np.concatenate([heightmap, satellite], axis=-1)
    else:
        raise ValueError(f'Unrecognised mode: {mode}')


def transform_back_output_img(img: np.ndarray, mode: str):
    if mode == 'SketchToHeightmap':
        return np.uint8(img * 127.5 + 127.5)
    elif mode in ['SketchToSatellite', 'HeightmapToSatellite', 'SketchHeightmapToSatellite']:
        return np.uint8(img * 255.0)


def sample_images(generator, input_img, out_image, step, mode):
    w_noise = np.random.normal(0, 1, (1, 14, 14, 1024))
    predicted = generator.predict([input_img[0:1], w_noise])[0]

    plotable_input_img = transform_back_input_img(input_img[0], mode)
    plotable_gen_img = transform_back_output_img(predicted, mode)
    plotable_target_img = transform_back_output_img(out_image[0], mode)

    # input image can be multi-channeled but each channels will always represent a separate map
    im_c = np.concatenate([input_img[0, ..., ch_idx] for ch_idx in range(plotable_input_img.shape[-1])], axis=-1)
    im_sat = np.concatenate([plotable_target_img, plotable_gen_img], axis=1).squeeze()

    plt.imsave('./outputs/heightmap_conversion' + 'sat' + str(step) + '.png', im_c)
    plt.imsave('./outputs/heightmap_conversion' + 'sat' + str(step) + '.png', im_sat)


def train_gan(x: np.ndarray, y: np.ndarray, mode, batch_size=20, n_epochs=100, ):
    bat_per_epo = int(len(x) / batch_size)
    n_steps = bat_per_epo * n_epochs
    avg_loss = 0
    avg_dloss = 0
    np.random.seed(42)
    tq_iter = tqdm(range(n_steps))

    for i in tq_iter:
        tq_iter.set_description(f"Step {i}", refresh=True)

        x_real, y_real, labels_real = generate_real_samples(x, y, batch_size, PATCH_SIZE)
        y_fake, labels_fake = generate_fake_samples(terrain_generator, x_real, PATCH_SIZE)
        w_noise = np.random.normal(0, 1, (batch_size, 14, 14, 1024))
        losses_composite = composite_model.train_on_batch([x_real, w_noise], [labels_real, y_real])
        loss_discriminator_fake = terrain_discriminator.train_on_batch([y_fake, x_real], labels_fake)
        loss_discriminator_real = terrain_discriminator.train_on_batch([y_real, x_real], labels_real)

        d_loss = (loss_discriminator_fake + loss_discriminator_real) / 2
        avg_dloss = avg_dloss + (d_loss - avg_dloss) / (i + 1)
        avg_loss = avg_loss + (losses_composite[0] - avg_loss) / (i + 1)
        print('total loss:' + str(avg_loss) + ' d_loss:' + str(avg_dloss))

        if i % 20 == 0:
            sample_images(terrain_generator, input_img=x_real, out_image=y_real, step=i, mode=mode)
        if i % 200 == 0:
            terrain_discriminator.save('terrain_discriminator_heightmap' + str(i) + '.h5', True)
            terrain_generator.save('terrain_generator_heightmap' + str(i) + '.h5', True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start Terrain GAN training ')
    parser.add_argument('--BN',
                        action='store_true',
                        help='Apply Batch Normalization')
    parser.add_argument('--SN',
                        action='store_true',
                        help='Apply Spectral Normalization to discriminator')
    parser.add_argument('--mode',
                        choices=['SketchToHeightmap', 'SketchToSatellite', 'HeightmapToSatellite',
                                 'SketchHeightmapToSatellite'],
                        help='Variation of GAN',
                        default='SketchToHeightmap',
                        metavar='')
    parser.add_argument('--learning_rate', default=0.00005,
                        help='optimizer learning rate')
    parser.add_argument('--beta', default=0.5,
                        help='optimizer beta parameter')
    parser.add_argument('--out_path',
                        help='root folder for samples and model save')

    args = parser.parse_args()
    data = np.load('training_data.npz')
    sketches = data['x']
    heightmaps = data['y']
    satellites = data['y_satelite']

    optd = Adam(learning_rate=0.00005, beta_1=0.5)
    builder = TerrainGANBuilder(spec_normalization=args.SN, batch_normalization=args.BN)
    terrain_generator, terrain_discriminator, composite_model = delegate_mode_building(builder, optd, args.mode)
    x_train, y_train = compile_training_data(sketches=sketches,
                                             heightmaps=heightmaps,
                                             satellites=satellites, mode=args.mode)

    composite_model.compile(loss=['binary_crossentropy', 'mae'], loss_weights=[1, 3], optimizer=optd)
    train_gan(x=x_train, y=y_train, mode=args.mode)
