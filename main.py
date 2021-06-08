import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from model import TerrainGANBuilder


def generate_real_samples(x, y_height_map, y_satellite, n_samples, patch_size):
    ix = np.random.randint(0, x.shape[0], n_samples)
    x_real = x[ix]
    heightmaps = y_height_map[ix]
    satelites = y_satellite[ix]
    y = np.ones((n_samples, patch_size, patch_size, 1))
    return x_real, y, heightmaps, satelites


def generate_fake_samples(generator, dataset, patch_size):
    w_noise = np.random.normal(0, 1, (dataset.shape[0], 14, 14, 1024))
    x = generator.predict([dataset, w_noise])
    y = np.zeros((len(x), patch_size, patch_size, 1))
    return x, y


def sample_images(generator, source, target_heightmap, target_satelite, idx):
    target_heightmap = np.uint8(target_heightmap * 127.5 + 127.5)
    target_satelite = np.uint8(target_satelite * 255)
    w_noise = np.random.normal(0, 1, (1, 14, 14, 1024))
    predicted = generator.predict([source, w_noise])
    im_heightmap = np.uint8(predicted[0, ...] * 127.5 + 127.5)
    im_satelite = np.uint8(predicted[0, ...] * 255)
    im_source = np.uint8(source[0, ...] * 255)
    print(im_source.shape)
    im_c = np.concatenate((im_heightmap[..., 0], np.squeeze(target_heightmap),
                           im_source[..., 0], im_source[..., 1], im_source[..., 2], im_source[..., 3]), axis=-1)
    im_sat = np.concatenate((im_satelite[:, :, 1:], np.squeeze(target_satelite)), axis=1)
    plt.imsave('./outputs/sketch_conversion' + str(idx) + '.png', im_c, cmap='terrain')
    plt.imsave('./outputs/sketch_conversion' + 'sat' + str(idx) + '.png', im_sat)


def train_gan(spectral_norm: bool = False, batch_norm: bool = False):
    data = np.load('training_data.npz')
    x_sketches = data['x']
    y_heightmaps = data['y']
    y_satellites = data['y_satelite']

    optd = Adam(learning_rate=0.00005, beta_1=0.5)
    builder = TerrainGANBuilder(spec_normalization=spectral_norm, batch_normalization=batch_norm)
    terrain_generator, terrain_discriminator, composite_model = builder.build_sketch_to_satelite(optd)
    composite_model.compile(loss=['binary_crossentropy', 'mae'], loss_weights=[1, 3], optimizer=optd)

    n_epochs, n_batch, = 100, 20
    bat_per_epo = int(len(x_sketches) / n_batch)
    patch_size = 15
    n_steps = bat_per_epo * n_epochs
    avg_loss = 0
    avg_dloss = 0
    np.random.seed(42)
    for i in tqdm(range(n_steps)):
        print('n_steps: {}'.format(i))
        x_real, labels_real, y_target_h, y_target_s = generate_real_samples(x_sketches,
                                                                            y_heightmaps,
                                                                            y_satellites,
                                                                            n_batch,
                                                                            patch_size)

        Y_target = np.concatenate([y_target_h, (y_target_s / 255.0)], axis=3)

        y_fake, labels_fake = generate_fake_samples(terrain_generator, x_real, patch_size)
        w_noise = np.random.normal(0, 1, (n_batch, 14, 14, 1024))
        losses_composite = composite_model.train_on_batch(
            [x_real, w_noise], [labels_real, Y_target])

        loss_discriminator_fake = terrain_discriminator.train_on_batch(
            [y_fake, x_real], labels_fake)
        loss_discriminator_real = terrain_discriminator.train_on_batch(
            [Y_target, x_real], labels_real)
        d_loss = (loss_discriminator_fake + loss_discriminator_real) / 2
        avg_dloss = avg_dloss + (d_loss - avg_dloss) / (i + 1)
        avg_loss = avg_loss + (losses_composite[0] - avg_loss) / (i + 1)
        print('total loss:' + str(avg_loss) + ' d_loss:' + str(avg_dloss))

        if i % 20 == 0:
            sample_images(terrain_generator, x_real[0:1, ...], y_target_h[0, ...], (y_target_s / 255.0)[0, ...], i)
        if i % 200 == 0:
            terrain_discriminator.save('terrain_discriminator' + str(i) + '.h5', True)
            terrain_generator.save('terrain_generator' + str(i) + '.h5', True)


if __name__ == '__main__':
    train_gan()
