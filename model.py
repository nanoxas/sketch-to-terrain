from keras.layers import *
from keras.models import Model
from keras.initializers import RandomNormal
import keras.backend as K


def UNet(shape):
    inputs = Input(shape)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    noise = Input((K.int_shape(conv5)[1], K.int_shape(conv5)[2], K.int_shape(conv5)[3]))
    conv5 = Concatenate()([conv5, noise])

    def get_scale_up(pr_input, out_channels):
        up6 = Conv2D(
            512,
            2,
            activation='relu',
            padding='same')(
            UpSampling2D(
                size=(
                    2,
                    2))(pr_input))
        merge6 = Concatenate()([conv4, up6])
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

        up7 = Conv2D(
            256,
            2,
            activation='relu',
            padding='same')(
            UpSampling2D(
                size=(
                    2,
                    2))(conv6))
        merge7 = Concatenate()([conv3, up7])
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

        up8 = Conv2D(
            128,
            2,
            activation='relu',
            padding='same')(
            UpSampling2D(
                size=(
                    2,
                    2))(conv7))
        merge8 = Concatenate()([conv2, up8])
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

        up9 = Conv2D(
            64,
            2,
            activation='relu',
            padding='same')(
            UpSampling2D(
                size=(
                    2,
                    2))(conv8))
        up9 = ZeroPadding2D(((0, 1), (0, 1)))(up9)
        merge9 = Concatenate()([conv1, up9])
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
        conv9 = Conv2D(32, 3, activation='relu', padding='same')(conv9)
        conv10 = Conv2D(out_channels, 1, activation='tanh')(conv9)
        return conv10

    satelite_output = get_scale_up(conv5, 3)
    highmap_output = get_scale_up(conv5, 1)
    c_out = Concatenate()([highmap_output, satelite_output])

    model = Model([inputs, noise], c_out)
    model.summary()
    return model


def patch_discriminator(shape):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=shape)
    cond_image = Input((225, 225, 4))
    conc_img = Concatenate()([in_image, cond_image])
    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same',
               kernel_initializer=init)(conc_img)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(128, (4, 4), strides=(2, 2),
               padding='same', kernel_initializer=init)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(256, (4, 4), strides=(2, 2),
               padding='same', kernel_initializer=init)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(512, (4, 4), strides=(2, 2),
               padding='same', kernel_initializer=init)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    x = LeakyReLU(alpha=0.2)(d)
    output = Conv2D(1, (4, 4), padding='same',
                    activation='sigmoid', kernel_initializer=init)(d)
    model = Model([in_image, cond_image], output)

    return model


def mount_discriminator_generator(g, d, image_shape):
    d.trainable = False
    input_gen = Input(shape=image_shape)
    input_noise = Input(shape=(14, 14, 1024))
    gen_out = g([input_gen, input_noise])
    output_d = d([gen_out, input_gen])
    model = Model(inputs=[input_gen, input_noise], outputs=[output_d, gen_out])
    model.summary()

    return model
