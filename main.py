from model import *
import numpy as np
from keras.optimizers import Adam, RMSprop
import matplotlib.pyplot as plt
from keras.models import load_model
from dataset_builder import *


def generate_real_samples(dataset, ground_trud_ds, n_samples, patch_size):
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    gt = ground_trud_ds[ix]
    y = np.ones((n_samples, patch_size, patch_size, 1))
    return X, y, gt


def generate_fake_samples(g_model, dataset, patch_size):
    w_noise = np.random.normal(0, 1, (dataset.shape[0], 14, 14, 1024))
    X = g_model.predict([dataset, w_noise])
    y = np.zeros((len(X), patch_size, patch_size, 1))
    return X, y


def sample_images(generator, source, target, idx):
    print(target.shape)
    target = np.uint8(target * 127.5 + 127.5)
    w_noise = np.random.normal(0, 1, (1, 14, 14, 1024))
    predicted = generator.predict([source, w_noise])
    im = np.uint8(predicted[0, ...] * 127.5 + 127.5)
    im_source = np.uint8(source[0, ...] * 255)
    print(im_source.shape)
    im_c = np.concatenate((np.squeeze(im, axis=-1), np.squeeze(target, axis=-1),
                           im_source[..., 0], im_source[..., 1], im_source[..., 2], im_source[..., 3]), axis=1)
    plt.imsave('./outputs/sketch_conversion' + str(idx) + '.png', im_c, cmap='terrain')


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
    input_shape_gen = (XTrain.shape[1], XTrain.shape[2], XTrain.shape[3])
    input_shape_disc = (YTrain.shape[1], YTrain.shape[2], YTrain.shape[3])

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
        X_real, labels_real, Y_target = generate_real_samples(XTrain, YTrain, n_batch, patch_size)
        Y_target[np.isnan(Y_target)] = 0
        X_real[np.isnan(X_real)] = 0

        Y_fake, labels_fake = generate_fake_samples(terrain_generator, X_real, patch_size)
        w_noise = np.random.normal(0, 1, (n_batch, 14, 14, 1024))
        losses_composite = composite_model.train_on_batch(
            [X_real, w_noise], [labels_real, Y_target])

        loss_discriminator_fake = terrain_discriminator.train_on_batch(
            [Y_fake, X_real], labels_fake)
        loss_discriminator_real = terrain_discriminator.train_on_batch(
            [Y_target, X_real], labels_real)
        d_loss = (loss_discriminator_fake + loss_discriminator_real) / 2
        avg_dloss = avg_dloss + (d_loss - avg_dloss) / (i + 1)
        avg_loss = avg_loss + (losses_composite[0] - avg_loss) / (i + 1)
        print('total loss:' + str(avg_loss) + ' d_loss:' + str(avg_dloss))

        if i % 100 == 0:
            sample_images(terrain_generator, X_real[0:1, ...], Y_target[0, ...], i)
        if i % 500 == 0:
            terrain_discriminator.save('terrain_discriminator' + str(i) + '.h5', True)
            terrain_generator.save('terrain_generator' + str(i) + '.h5', True)


if __name__ == '__main__':
    # extract_patches_from_raster()
    # compute_sketches()
    train_gan()
    # test_gan()
