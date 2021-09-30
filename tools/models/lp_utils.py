"""
Utility functions used in tools

Author: Venkanna Babu Guthula
Date: 03-07-2021
"""

import csv
import sys
from models import resunet_model, unet_model, segnet_model, unet_mini  # FCNs
import numpy as np
from keras import applications


# Load model
def select_model(args):
    if args.model == "unet":
        model = unet_model.unet(args)
        if args.weights:
            model.load_weights(args.weights)
    elif args.model == "resunet":
        model = resunet_model.build_res_unet(args)
        if args.weights:
            model.load_weights(args.weights)
    elif args.model == "segnet":
        model = segnet_model.create_segnet(args)
        if args.weights:
            model.load_weights(args.weights)
    elif args.model == "unet_mini":
        model = unet_mini.UNet(args)
        if args.weights:
            model.load_weights(args.weights)
        return model

    # CNN Models (https://keras.io/api/applications)
    elif args.model == "vgg16":
        model = applications.vgg16.VGG16(weights=None, input_shape=args.input_shape, classes=args.num_classes)
        if args.weights:
            model.load_weights(args.weights)
    elif args.model == "resnet50":
        model = applications.resnet50.ResNet50(weights=None, input_shape=args.input_shape, classes=args.num_classes)
        if args.weights:
            model.load_weights(args.weights)
    elif args.model == "xception":
        model = applications.xception.Xception(weights=None, input_shape=args.input_shape, classes=args.num_classes)
        if args.weights:
            model.load_weights(args.weights)
    elif args.model == "nasnet":
        model = applications.nasnet.NASNetLarge(weights=None, input_shape=args.input_shape, classes=args.num_classes)
        if args.weights:
            model.load_weights(args.weights)
    else:
        print(args.model + "Model does not exist, select model from"
                           " unet, unet_mini, resunet, segnet, vgg16, resnet50, xception, nasnet")
        sys.exit()

    return model


# Get image and label paths from csv file
def file_paths(csv_file):
    image_paths = []
    label_paths = []
    with open(csv_file, 'r', newline='\n') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            image_paths.append(row[0])
            label_paths.append(row[1])
    return image_paths, label_paths


# Find the radiometric resolution of an image and get max number of an image
# To rescale the input data
def rescaling_value(value):
    return pow(2, value) - 1


def deep_lulc_data(array):
    # converting deep globe lulc labels to single band array
    # https://competitions.codalab.org/competitions/18468#participate-get_starting_kit
    array = np.where(array > 128, 1, 0)
    array1 = array[0]
    array2 = array[1] * 2
    array3 = array[2] * 4
    array = array1 + array2 + array3
    final_array = np.where(array == 7, 1, array)
    print(np.unique(final_array))
    return final_array
