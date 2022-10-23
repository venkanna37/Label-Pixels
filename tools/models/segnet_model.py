"""
SegNet Model
 - refer: https://arxiv.org/abs/1511.00561
"""

from typing import Tuple, List, Text, Dict, Any, Iterator
import numpy as np
import h5py
from keras.engine.training import Model as tModel
from keras.models import Model
from keras import layers
from keras.layers import Input
from keras.layers.core import Activation, Dropout
from keras.layers.convolutional import Conv2D, UpSampling2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Layer
from keras.backend import argmax, gradients, sum, repeat_elements


class DePool2D(UpSampling2D):
    '''
    https://github.com/nanopony/keras-convautoencoder/blob/c8172766f968c8afc81382b5e24fd4b57d8ebe71/autoencoder_layers.py#L24
    Simplar to UpSample, yet traverse only maxpooled elements.
    '''

    def __init__(self, pool2d_layer: MaxPooling2D, *args, **kwargs):
        self._pool2d_layer = pool2d_layer
        super().__init__(*args, **kwargs)

    def get_output(self, train: bool=False) -> Any:
        X = self.get_input(train)
        if self.dim_ordering == 'th':
            output = repeat_elements(X, self.size[0], axis=2)
            output = repeat_elements(output, self.size[1], axis=3)
        elif self.dim_ordering == 'tf':
            output = repeat_elements(X, self.size[0], axis=1)
            output = repeat_elements(output, self.size[1], axis=2)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        return gradients(
            sum(
                self._pool2d_layer.get_output(train)
            ),
            self._pool2d_layer.get_input(train)
        ) * output


def write_new_VGG_weights(input_shape, oldfname, outfname):
    """Write weights of reshaped VGG."""
    nfeatures = input_shape[2]
    with h5py.File(oldfname, "r") as f:
        # print(list(f["block1_conv1"]["block1_conv1_W_1:0"].keys()))
        W = np.array(f["block1_conv1"]["block1_conv1_W_1:0"])
    W_padded = np.random.randn(W.shape[0],
                              W.shape[1],
                              nfeatures,
                              W.shape[3])
    # randomly initialize using original filters
    for i in range(nfeatures):
        nf_idx = np.random.randint(0, W.shape[2])
        W_padded[:, :, i, :] = W[:, :, nf_idx, :]
    with h5py.File(outfname, "a") as f:
        del f["block1_conv1"]["block1_conv1_W_1:0"]
        f["block1_conv1"]["block1_conv1_W_1:0"] = W_padded
    return outfname


def VGG16_encoder(input_shape, init=True):
    """Creates a VGG16 encoder with reshaped input dimensions."""
    root_dir = "./models/"
    input_tensor = Input(shape=input_shape)
    if init:
        weights_fname = root_dir + "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    if init and input_shape[2] != 3:
        outfname = root_dir + "vgg16_with_" + str(input_shape[2]) + "_bands.h5"
        weights_fname = write_new_VGG_weights(input_shape, weights_fname, outfname)

    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(input_tensor)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    model = Model(input_tensor, x, name="vgg16")
    if init:
        model.load_weights(weights_fname)

    return model


def create_segnet(input_shape, num_classes, indices=True, ker_init="he_normal") -> tModel:

    # input_shape = tuple(args.input_shape)
    if input_shape[2] == 3:
        init = True
    else:
        init = False
    encoder = VGG16_encoder(input_shape, init=False)

    L = [layer for i, layer in enumerate(encoder.layers)] # type: List[Layer]
    #for layer in L: layer.trainable = False # freeze VGG16
    L.reverse()

    x = encoder.output
    x = Dropout(0.5)(x)
    # Block 5
    if indices: x = DePool2D(L[0], size=L[0].pool_size, input_shape=encoder.output_shape[1:])(x)
    else:       x = UpSampling2D(  size=L[0].pool_size, input_shape=encoder.output_shape[1:])(x)
    x = Activation('relu')(BatchNormalization()(Conv2D(L[1].filters, L[1].kernel_size, padding=L[1].padding, kernel_initializer=ker_init)(x)))
    x = Activation('relu')(BatchNormalization()(Conv2D(L[2].filters, L[2].kernel_size, padding=L[2].padding, kernel_initializer=ker_init)(x)))
    x = Activation('relu')(BatchNormalization()(Conv2D(L[3].filters, L[3].kernel_size, padding=L[3].padding, kernel_initializer=ker_init)(x)))
    x = Dropout(0.5)(x)
    # Block 4
    if indices: x = DePool2D(L[4], size=L[4].pool_size)(x)
    else:       x = UpSampling2D(  size=L[4].pool_size)(x)
    x = Activation('relu')(BatchNormalization()(Conv2D(L[5].filters, L[5].kernel_size, padding=L[5].padding, kernel_initializer=ker_init)(x)))
    x = Activation('relu')(BatchNormalization()(Conv2D(L[6].filters, L[6].kernel_size, padding=L[6].padding, kernel_initializer=ker_init)(x)))
    x = Activation('relu')(BatchNormalization()(Conv2D(L[7].filters, L[7].kernel_size, padding=L[7].padding, kernel_initializer=ker_init)(x)))
    x = Dropout(0.5)(x)
    # Block 3
    if indices: x = DePool2D(L[8], size=L[8].pool_size)(x)
    else:       x = UpSampling2D(  size=L[8].pool_size)(x)
    # x = ZeroPadding2D(padding=(0, 1))(x)
    x = Activation('relu')(BatchNormalization()(Conv2D(L[10].filters, L[10].kernel_size, padding=L[10].padding, kernel_initializer=ker_init)(x)))
    x = Activation('relu')(BatchNormalization()(Conv2D(L[11].filters, L[11].kernel_size, padding=L[11].padding, kernel_initializer=ker_init)(x)))
    x = Dropout(0.5)(x)
    # Block 2
    if indices: x = DePool2D(L[12], size=L[12].pool_size)(x)
    else:       x = UpSampling2D(   size=L[12].pool_size)(x)
    x = Activation('relu')(BatchNormalization()(Conv2D(L[13].filters, L[13].kernel_size, padding=L[13].padding, kernel_initializer=ker_init)(x)))
    x = Activation('relu')(BatchNormalization()(Conv2D(L[14].filters, L[14].kernel_size, padding=L[14].padding, kernel_initializer=ker_init)(x)))
    # Block 1
    if indices: x = DePool2D(L[15], size=L[15].pool_size)(x)
    else:       x = UpSampling2D(size=L[15].pool_size)(x)
    x = Activation('relu')(BatchNormalization()(Conv2D(L[16].filters, L[16].kernel_size, padding=L[16].padding, kernel_initializer=ker_init)(x)))
    x = Activation('relu')(BatchNormalization()(Conv2D(L[17].filters, L[17].kernel_size, padding=L[17].padding, kernel_initializer=ker_init)(x)))

    x = Conv2D(num_classes, (1, 1), padding='valid', kernel_initializer=ker_init)(x)

    if num_classes == 1:
        x = Activation('sigmoid')(x)
    elif num_classes > 1:
        x = Activation('softmax')(x)
    
    predictions = x

    segnet = Model(inputs=encoder.inputs, outputs=predictions)  # type: tModel

    return segnet


def model_summary(args):
    model = create_segnet(args)
    model.summary()

