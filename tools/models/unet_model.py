"""
UNet model
"""

import tensorflow.keras as keras
import argparse


# Model
def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p


# Passing arguments for commands line
def add__parser(subparser):
    parser = subparser.add_parser("unet_summary", help="UNet Model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_shape", nargs='+', type=int,  help="Input shape of the data (rows, columns, channels)")
    parser.set_defaults(func=unet_model.model_summary)


def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c


def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c


def unet(args):
    input_shape = tuple(args.input_shape)
    f = [64, 128, 256, 512, 1024]
    inputs = keras.layers.Input(input_shape)

    p0 = inputs
    c1, p1 = down_block(p0, f[0])  # 128
    c2, p2 = down_block(p1, f[1])  # 64 -> 32
    c3, p3 = down_block(p2, f[2])  # 32 -> 16
    c4, p4 = down_block(p3, f[3])  # 16->8

    bn = bottleneck(p4, f[4])

    u1 = up_block(bn, c4, f[3])  # 8 -> 16
    u2 = up_block(u1, c3, f[2])  # 16 -> 32
    u3 = up_block(u2, c2, f[1])  # 32 -> 64
    u4 = up_block(u3, c1, f[0])  # 64 -> 128

    if args.num_classes > 1:
        outputs = keras.layers.Conv2D(args.num_classes, (1, 1), padding="same", activation="softmax")(u4)
    elif args.num_classes == 1:
        outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
    model = keras.models.Model(inputs, outputs)
    return model
