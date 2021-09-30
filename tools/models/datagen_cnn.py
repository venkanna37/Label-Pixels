"""
Preparing input data (X and Y) to feed networks

Author: Venkanna Babu Guthula
Date: 30-09-2021

Limitation: Currently works with only CNN
"""

import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from tensorflow.keras.applications.nasnet import preprocess_input as preprocess_input_nasnet
from tensorflow.keras.applications.xception import preprocess_input as preprocess_input_xception


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, image_paths, label_paths, batch_size=4, n_classes=2, patch_size=224, shuffle=True, net="vgg16"):
        'Initialization'
        self.batch_size = batch_size
        self.label_paths = label_paths
        self.image_paths = image_paths
        self.n_classes = n_classes
        self.patch_size = patch_size
        self.shuffle = shuffle
        self.net = net
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        n_classes = self.n_classes
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        list_image_temp = [self.image_paths[k] for k in indexes]
        list_label_temp = [self.label_paths[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_image_temp, list_label_temp, self.n_classes, self.patch_size, self.net)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, image_paths, label_paths, n_classes, patch_size,  net):
        # Initialization
        X = np.zeros(shape=(0, patch_size, patch_size, 3))  # assuming input is 3 (RGB) channels
        y = []
        # Generate data
        for img, label in zip(image_paths, label_paths):
            # Store sample
            _image = image.load_img(img, target_size=(patch_size, patch_size))
            _image = image.img_to_array(_image)
            _image = np.expand_dims(_image, axis=0)
            if net == "vgg16":
                _image = preprocess_input_vgg16(_image)
            elif net == "resnet50":
                _image = preprocess_input_resnet50(_image)
            elif net == "xception":
                _image = preprocess_input_xception(_image)
            elif net == "nasnet":
                _image = preprocess_input_nasnet(_image)
            X = np.concatenate((X, _image), axis=0)
            _y = int(label)
            _y = to_categorical(_y, n_classes)
            y.append(_y)

        y = np.array(y)
        return X, y
