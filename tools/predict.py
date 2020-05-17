import csv
import gdal
import numpy as np
import os
from tensorflow.keras.models import load_model
import argparse
from models import segnet_model


def predict(args):
    if args.model == "unet":
        model = load_model(args.weights)
    elif args.model == "resunet":
        model = load_model(args.weights)
    elif args.model == "segnet":
        model = segnet_model.create_segnet(args)
        model.load_weights(args.weights)

    paths_file = args.csv_paths
    test_image_paths = []
    test_label_paths = []
    with open(paths_file, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            test_image_paths.append(row[0])
            test_label_paths.append(row[1])
    outdriver = gdal.GetDriverByName("GTiff")

    for i in range(len(test_image_paths)):
        image = gdal.Open(test_image_paths[i])
        image_array = np.array(image.ReadAsArray())/255
        _, __, patch_size = image_array.shape
        image_array = image_array.transpose(1, 2, 0)
        image_array = np.expand_dims(image_array, axis=0)
        result_array = model.predict(image_array)
        result_array = np.reshape(result_array[0], (256, 256))
        outfile = args.output_folder + os.path.basename(test_image_paths[i])
        outdata = outdriver.Create(str(outfile), patch_size, patch_size, 1, gdal.GDT_Float64)
        outdata.GetRasterBand(1).WriteArray(result_array)
        print("Predicted " + str(i+1) + " Images")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Name of the model, should be from unet, resunet, segnet")
    parser.add_argument("--input_shape", nargs='+', type=int, help="Input shape of the model (rows, columns, channels)")
    parser.add_argument("--weights", type=str, help="Name and path of the trained model")
    parser.add_argument("--csv_paths", type=str, help="CSV file with image and label paths")
    parser.add_argument("--output_folder", type=str, help="Folder to save the predicted images")
    args = parser.parse_args()
    predict(args)
