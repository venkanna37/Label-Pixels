"""
Preparing input data to feed networks
"""
import numpy as np
import tensorflow.keras as keras
import gdal
from tensorflow.keras.utils import to_categorical


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, image_paths, label_paths, batch_size=32, n_classes=2, n_channels=3, patch_size=128,
                 shuffle=True, rs=255, rs_label=1):
        'Initialization'
        self.batch_size = batch_size
        self.label_paths = label_paths
        self.image_paths = image_paths
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.shuffle = shuffle
        self.rescale_value = rs
        self.rs_label = rs_label
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
        X, y = self.__data_generation(list_image_temp, list_label_temp, n_classes, self.rescale_value, self.rs_label)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, image_paths, label_paths, n_classes, rescale_value, rs_label):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = []
        y = []
        # Generate data
        for image, label in zip(image_paths, label_paths):
            # Store sample
            _image = gdal.Open(image)
            _label = gdal.Open(label)
            _image = np.array(_image.ReadAsArray()) / rescale_value
            _image = _image.transpose(1, 2, 0)
            _label = np.array(_label.ReadAsArray()) / rs_label
            _label = np.expand_dims(_label, axis=-1)
            # print(_label.shape)
            if n_classes > 1:
                _label = to_categorical(_label, num_classes=n_classes)
                # print(_label.shape)
            X.append(_image)
            y.append(_label)
        X = np.array(X)
        y = np.array(y)
        # print(X.shape, y.shape)
        return X, y
