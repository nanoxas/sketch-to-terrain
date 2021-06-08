from typing import Dict, Tuple

import tensorflow
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (
    BatchNormalization, Concatenate, Conv2D, Input, Layer, LeakyReLU, MaxPooling2D, ReLU, UpSampling2D, ZeroPadding2D)
from tensorflow.keras.models import Model
from tensorflow_addons.layers import SpectralNormalization


class TerrainGANBuilder:
    def __init__(self, spec_normalization: bool = False, batch_normalization: bool = False):
        self.spec_normalization = spec_normalization
        self.batch_normalization = batch_normalization
        self.disc_init_fn = RandomNormal(stddev=0.02)
        self._map_shape = (255, 255)
        self._unet_repr_size = (14, 14, 1024)

    def build_sketch_to_terrain(self) -> Tuple[tensorflow.keras.Model, ...]:
        """
        Initial model. From sketches generate heightmaps.
        :return: complete
        """
        gen_image_shape = self._channels_shape(4)
        gen_out_shape = self._channels_shape(1)
        downsampling_outs, gen_inputs = self._get_scale_down(gen_image_shape)
        upsampling_out = self._get_scale_up(downsampling_outs, 1)

        generator = Model([gen_inputs['image_input'], gen_inputs['noise']], upsampling_out)
        discriminator = self._patch_discriminator(gen_out_shape)

        # TODO: explain why this is not trainable
        discriminator.trainable = False
        input_gen = Input(shape=gen_image_shape)
        input_noise = Input(shape=self._unet_repr_size)
        gen_out = generator([input_gen, input_noise])
        output_d = discriminator([gen_out, input_gen])
        full_gan = Model(inputs=[input_gen, input_noise], outputs=[output_d, gen_out])

        return generator, discriminator, full_gan

    def build_sketch_to_satelite(self, optimizer, sequential: bool = True):
        """
        From sketches generate heightmaps *and* satellites.
        :param sequential: If True, satellites will be generated based on information from generated heightmap and
        sketches, otherwise only sketches will be used. Please refer to the README.md for more information on
        architecture
        :return:
        """
        """
        Initial model. From sketches generate heightmaps.
        :return: complete
        """
        gen_image_shape = self._channels_shape(4)
        gen_out_shape = self._channels_shape(4)  # 1 channel for heightmap, 3 channels for satellites
        downsampling_outs, gen_inputs = self._get_scale_down(gen_image_shape)

        upsampling_heightmap_out = self._get_scale_up(downsampling_outs, 1)
        upsampling_satellite_out = self._get_scale_up(downsampling_outs, 3)

        concat_out = Concatenate()([upsampling_heightmap_out, upsampling_satellite_out])

        generator = Model([gen_inputs['image_input'], gen_inputs['noise']], concat_out)
        discriminator = self._patch_discriminator(gen_out_shape)

        return self._mount_single(optimizer, generator, discriminator, gen_image_shape)

    def build_terrain_to_satelite(self):
        raise NotImplementedError

    def _mount_single(self, optimizer, generator, discriminator, gen_image_shape):
        discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
        # TODO: explain why trainable False
        discriminator.trainable = False
        input_gen = Input(shape=gen_image_shape)
        input_noise = Input(shape=self._unet_repr_size)
        gen_out = generator([input_gen, input_noise])
        output_d = discriminator([gen_out, input_gen])
        full_gan = Model(inputs=[input_gen, input_noise], outputs=[output_d, gen_out])
        return generator, discriminator, full_gan

    def _channels_shape(self, channels: int) -> Tuple[int, ...]:
        return *self._map_shape, channels

    def _add_conv_layer(self, layer: Layer, x) -> Layer:
        out = layer(x)
        if self.batch_normalization:
            out = BatchNormalization()(out)
        return ReLU(out)

    def _get_scale_down_block(self, previous_output, filter_out: int):
        conv = self._add_conv_layer(Conv2D(filter_out, 3, padding='same'), previous_output)
        conv = self._add_conv_layer(Conv2D(filter_out, 3, padding='same'), conv)
        return conv, MaxPooling2D(pool_size=(2, 2))(conv)

    def _get_scale_up_block(self, previous_output, combine_with, filter_out: int):
        upsampled = UpSampling2D(size=(2, 2))(previous_output)
        up = self._add_conv_layer(Conv2D(filter_out, 2, padding='same'), upsampled)
        merge = Concatenate()([combine_with, up])
        conv = Conv2D(filter_out, 3, padding='same')(merge)
        return Conv2D(filter_out, 3, padding='same')(conv)

    def _discriminator_block(self, previous_output, filter_out: int, strides: Tuple[int, int] = (2, 2)):
        layer = Conv2D(filter_out, (4, 4), strides=strides, padding='same', kernel_initializer=self.disc_init_fn)
        if self.spec_normalization:
            layer = SpectralNormalization(layer)
        out = layer(previous_output)
        if self.batch_normalization:
            out = BatchNormalization()(out)
        return LeakyReLU(alpha=0.2)(out)

    def _get_scale_down(self, shape: Tuple[int, ...]) -> Tuple[Dict[str, Layer], ...]:
        """
        Function to calculate first downsampling branch (down conv).
        @Parameters:
        :params inputs - shape of training data
        :return: dict with graph inputs, dict with conv outs
        """
        inputs = Input(shape=shape)
        conv1, pool1 = self._get_scale_down_block(inputs, 64)
        conv2, pool2 = self._get_scale_down_block(pool1, 128)
        conv3, pool3 = self._get_scale_down_block(pool2, 256)
        conv4, pool4 = self._get_scale_down_block(pool3, 512)
        conv5 = self._add_conv_layer(Conv2D(1024, 3, padding='same'), pool4)
        conv5 = self._add_conv_layer(Conv2D(1024, 3, padding='same'), conv5)
        noise = Input((K.int_shape(conv5)[1], K.int_shape(conv5)[2], K.int_shape(conv5)[3]))
        conv5 = Concatenate()([conv5, noise])

        conv_out = {
            'conv1': conv1,
            'conv2': conv2,
            'conv3': conv3,
            'conv4': conv4,
            'conv5': conv5
        }

        ins = {
            'image_input': inputs,
            'noise': noise
        }

        return ins, conv_out

    def _get_scale_up(self, downsampling_outs: Dict[str, Layer], out_channels: int):
        block1 = self._get_scale_up_block(downsampling_outs['conv5'], downsampling_outs['conv4'], 512)
        block2 = self._get_scale_up_block(block1, downsampling_outs['conv3'], 256)
        block3 = self._get_scale_up_block(block2, downsampling_outs['conv2'], 128)
        up9 = Conv2D(64, 2, padding='same')(UpSampling2D(size=(2, 2))(block3))
        up9 = ZeroPadding2D(((0, 1), (0, 1)))(up9)
        merge9 = Concatenate()([downsampling_outs['conv1'], up9])
        conv9 = Conv2D(64, 3, padding='same')(merge9)
        conv9 = Conv2D(64, 3, padding='same')(conv9)
        conv9 = Conv2D(32, 3, padding='same')(conv9)
        conv10 = Conv2D(out_channels, 1, activation='tanh')(conv9)
        return conv10

    def _patch_discriminator(self, shape: Tuple[int, ...]) -> Model:
        in_image = Input(shape=shape)
        cond_image = Input((225, 225, 4))
        conc_img = Concatenate()([in_image, cond_image])

        block1 = self._discriminator_block(conc_img, 64)
        block2 = self._discriminator_block(block1, 128)
        block3 = self._discriminator_block(block2, 256)
        block4 = self._discriminator_block(block3, 512)
        block5 = self._discriminator_block(block4, 512, strides=(1, 1))
        final_layer = Conv2D(1, (4, 4), padding='same', activation='sigmoid', kernel_initializer=self.disc_init_fn)
        if self.spec_normalization:
            final_layer = SpectralNormalization(final_layer)
        output = final_layer(block5)
        return Model([in_image, cond_image], output)
