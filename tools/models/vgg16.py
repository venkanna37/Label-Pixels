"""
VGG16 model
Paper:
The below article referred to write the code
https://towardsdatascience.com/creating-vgg-from-scratch-using-tensorflow-a998a5640155

Author: Venkanna Babu Guthula
Date: 03-07-2021
"""

# import necessary layers
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.layers import MaxPool2D, Flatten, Dense
from tensorflow.keras import Model


def vgg16(args):
    # input
    inputs = Input((224, 224, 3))
    # 1st Conv Block
    x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    # 2nd Conv Block
    x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    # 3rd Conv block
    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    # 4th Conv block
    x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    # 5th Conv block
    x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    # Fully connected layers
    x = Flatten()(x)
    x = Dense(units=4096, activation='relu')(x)
    x = Dense(units=4096, activation='relu')(x)

    if args.num_classes > 1:
        outputs = Dense(units=args.num_classes, activation='softmax')(x)
    elif args.num_classes == 1:
        outputs = Dense(units=1, activation='sigmoid')(x)  # Need to check and fixed
    model = Model(inputs, outputs)

    return model
