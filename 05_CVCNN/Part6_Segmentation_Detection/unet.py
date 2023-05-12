import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential, layers

BATCH_SIZE = 32
NUM_LABELS = 1
WIDTH = 512
HEIGHT = 512

class convBlock(layers.Layer):
    def __init__(self, out_ch, padding='same', kernel_size=3):
        super().__init__()
        kernel_size = kernel_size
        pad_size = lambda kernel_size: (kernel_size-1)//2

        self.conv_1 = layers.Conv2D(out_ch, (kernel_size, kernel_size),
                                    strides=(1, 1), padding=padding)
        self.relu = layers.Activation('relu')

        self.conv_2 = layers.Conv2D(out_ch, (kernel_size, kernel_size),
                                    strides=(1, 1), padding=padding)

#         self.INorm = tfa.layers.InstanceNormalization(axis=3,
#                                                       center=True,
#                                                       scale=True)

    def call(self, input, training=None):
        x = self.conv_1(input)
#         x = self.INorm(x)
        x = self.relu(x)
        x = self.conv_2(x)
#         x = self.INorm(x)
        x = self.relu(x)
        return x

class Encoder(layers.Layer):
    def __init__(self, chs=(32, 64, 128, 256, 512), padding='same'):
        super().__init__()
        self.FPN_enc_ftrs = [convBlock(chs[i]) for i in range(len(chs))]
        self.pool = layers.MaxPooling2D(pool_size=(2, 2),
                                        strides=(2, 2), padding=padding)

    def call(self, x, training=None):
        features = []
        for block in self.FPN_enc_ftrs:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return features


class UpSampleConvs(layers.Layer):
    def __init__(self, out_ch, padding='same'):
        super().__init__()
        self.conv = layers.Conv2D(out_ch, (3, 3),
                                  strides=(1, 1), padding=padding)
        self.relu = layers.Activation('relu')
        self.upSample = layers.UpSampling2D(size=2)
#         self.INorm = tfa.layers.InstanceNormalization(axis=3,
#                                                       center=True,
#                                                       scale=True)

    def call(self, x):
        x = self.upSample(x)
        x = self.conv(x)
        # x = self.INorm(x)
        x = self.relu(x)
        return x


class Decoder(layers.Layer):
    def __init__(self, chs=(256, 128, 64, 32), padding='same'):
        super().__init__()

        self.chs = chs
        self.padding = padding
        # 上採樣後卷積
        self.upconvs = [UpSampleConvs(chs[i], padding=padding)
                        for i in range(len(chs))]
        self.FPN_dec_ftrs = [convBlock(chs[i], padding=padding)
                             for i in range(len(chs))]

    def call(self, x, encoder_features):
        for i in range(len(self.chs)):
            enc_ftrs = encoder_features[i]
            x = self.upconvs[i](x)

            # enc_ftrs = self.crop(encoder_features[i], x)
            x = layers.Concatenate(axis=-1)([x, enc_ftrs])
            x = self.FPN_dec_ftrs[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, H, W, _ = x.shape
        enc_ftrs = layers.CenterCrop(H, W)(enc_ftrs)
        return enc_ftrs


class UNet(Model):
    def __init__(self, enc_chs=(64, 128, 256, 512, 1024),
                 dec_chs=(512, 256, 128, 64),
                 num_class=1, padding='same',
                 retain_dim=None, activation=None):
        super().__init__()
        self.encoder = Encoder(enc_chs, padding=padding)
        self.decoder = Decoder(dec_chs, padding=padding)
        self.head = layers.Conv2D(num_class, (1, 1),
                                  strides=(1, 1), padding=padding)
        self.retain_dim = retain_dim
        self.activation = activation

    def call(self, inputs):
        enc_ftrs = self.encoder(inputs)
        # 把不同尺度的所有featuremap都輸入decoder，我們在decoder需要做featuremap的拼接
        outputs = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        outputs = self.head(outputs)

        if self.retain_dim:
            outputs = tf.image.resize(outputs,
                                      self.retain_dim,
                                      method='nearest')

        if self.activation:
            outputs = self.activation(outputs)

        return outputs

