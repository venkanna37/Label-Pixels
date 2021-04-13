"""
List of functions used in tools
"""

import csv
import sys
from models import resunet_model, unet_model, segnet_model, unet_mini


# Load model
def select_model(args):
    if args.model == "unet":
        model = unet_model.unet(args)
        if args.weights:
            model.load_weights(args.weights)
        return model
    elif args.model == "resunet":
        model = resunet_model.build_res_unet(args)
        if args.weights:
            model.load_weights(args.weights)
        return model
    elif args.model == "segnet":
        model = segnet_model.create_segnet(args)
        if args.weights:
            model.load_weights(args.weights)
        return model
    elif args.model == "unet_mini":
        model = unet_mini.UNet(args)
        if args.weights:
            model.load_weights(args.weights)
        return model
    else:
        print(args.model + "Model does not exist, select model from"
                           " unet, unet_mini, resunet and segnet")
        sys.exit()


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


