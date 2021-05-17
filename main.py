from model import *
import numpy as np
from keras.optimizers import Adam, RMSprop
import matplotlib.pyplot as plt
from keras.models import load_model
from dataset_builder import *


def generate_real_samples(X, y_height_map, y_satellite, n_samples, patch_size):
    ix = np.random.randint(0, X.shape[0], n_samples)
    x_real = X[ix]
    heightmaps = y_height_map[ix]
    satelites = y_satellite[ix]
    y = np.ones((n_samples, patch_size, patch_size, 1))
    return x_real, y, heightmaps, satelites


def generate_fake_samples(g_model, dataset, patch_size):
    w_noise = np.random.normal(0, 1, (dataset.shape[0], 14, 14, 1024))
    X = g_model.predict([dataset, w_noise])
    y = np.zeros((len(X), patch_size, patch_size, 1))
    return X, y


def sample_images(generator, source, target_heightmap, target_satelite, idx):
    print(target_heightmap.shape)
    target_heightmap = np.uint8(target_heightmap * 127.5 + 127.5)
    w_noise = np.random.normal(0, 1, (1, 14, 14, 1024))
    predicted = generator.predict([source, w_noise])
    im = np.uint8(predicted[0, ...] * 127.5 + 127.5)
    im_source = np.uint8(source[0, ...] * 255)
    print(im_source.shape)
    im_c = np.concatenate((im[..., 0], np.squeeze(target_heightmap),
                           im_source[..., 0], im_source[..., 1], im_source[..., 2], im_source[..., 3]), axis=-1)
    im_sat = np.concatenate((np.uint8(predicted.squeeze()[:, :, 1:] * 255), target_satelite), axis=1)
    plt.imsave('./outputs/sketch_conversion' + str(idx) + '.png', im_c, cmap='terrain')
    plt.imsave('./outputs/sketch_conversion' + 'sat' + str(idx) + '.png', im_sat)


def test_gan():
    terrain_generator = load_model('terrain_generator26500.h5')
    data = np.load('training_data.npz')
    XTrain = data['x']
    YTrain = data['y']
    for i in range(200):
        source = XTrain[i:i + 1, ...]
        target = YTrain[i, ...]
        sample_images(terrain_generator, source, target, i)


def train_gan():
    data = np.load('training_data.npz')
    XTrain = data['x']
    YTrain = data['y']
    YTrain_satelite = data['y_satelite']
    input_shape_gen = (XTrain.shape[1], XTrain.shape[2], XTrain.shape[3])
    input_shape_disc = (YTrain.shape[1], YTrain.shape[2], YTrain.shape[3] + YTrain_satelite.shape[3])

    terrain_generator = UNet(input_shape_gen)
    terrain_discriminator = patch_discriminator(input_shape_disc)
    optd = Adam(0.0001, 0.5)
    terrain_discriminator.compile(loss='binary_crossentropy', optimizer=optd)
    composite_model = mount_discriminator_generator(
        terrain_generator, terrain_discriminator, input_shape_gen)
    composite_model.compile(
        loss=[
            'binary_crossentropy', 'mae'], loss_weights=[
            1, 3], optimizer=optd)

    n_epochs, n_batch, = 100, 20
    bat_per_epo = int(len(XTrain) / n_batch)
    patch_size = 15
    n_steps = bat_per_epo * n_epochs
    min_loss = 999
    avg_loss = 0
    avg_dloss = 0
    for i in range(n_steps):
        X_real, labels_real, Y_target_h, Y_target_s = generate_real_samples(XTrain, YTrain, YTrain_satelite, n_batch,
                                                                            patch_size)
        Y_target_h[np.isnan(Y_target_h)] = 0
        X_real[np.isnan(X_real)] = 0
        Y_target = np.concatenate([Y_target_h, Y_target_s], axis=-1)

        Y_fake, labels_fake = generate_fake_samples(terrain_generator, X_real, patch_size)
        w_noise = np.random.normal(0, 1, (n_batch, 14, 14, 1024))
        losses_composite = composite_model.train_on_batch(
            [X_real, w_noise], [labels_real, Y_target_h])

        loss_discriminator_fake = terrain_discriminator.train_on_batch(
            [Y_fake, X_real], labels_fake)
        loss_discriminator_real = terrain_discriminator.train_on_batch(
            [Y_target, X_real], labels_real)
        d_loss = (loss_discriminator_fake + loss_discriminator_real) / 2
        avg_dloss = avg_dloss + (d_loss - avg_dloss) / (i + 1)
        avg_loss = avg_loss + (losses_composite[0] - avg_loss) / (i + 1)
        print('total loss:' + str(avg_loss) + ' d_loss:' + str(avg_dloss))

        if i % 100 == 0:
            sample_images(terrain_generator, X_real[0:1, ...], Y_target_h[0, ...], YTrain_satelite[0, ...], i)
        if i % 500 == 0:
            terrain_discriminator.save('terrain_discriminator' + str(i) + '.h5', True)
            terrain_generator.save('terrain_generator' + str(i) + '.h5', True)


if __name__ == '__main__':
    # TODO: clutch into sequential steps
    # recombine_heightmap_sattelite_data()
    # extract_patch_simple_map()

    # extract_patches_from_raster()
    # compute_sketches()
    train_gan()
    # test_gan()
