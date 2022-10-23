"""
predicts every patch and save as tif file
Working well for multi-class  classification (softmax)
  - Tested with unet_mini model
working well for binary classification (sigmoid)
  - Tested with resunet model
"""

import gdal
import numpy as np
import os
import argparse
from models import lp_utils as lu


def predict(model_name, weights_path, input_shape, num_classes, csv_paths, rs, output_folder):
    # Loading model with weights
    model = lu.select_model(model_name, weights_path, input_shape, num_classes)
    # Saving image paths as a list
    test_image_paths, test_label_paths = lu.file_paths(csv_paths)
    # Predicting patches
    outdriver = gdal.GetDriverByName("GTiff")
    rescale_value = lu.rescaling_value(rs)
    for i in range(len(test_image_paths)):
        image = gdal.Open(test_image_paths[i])
        image_array = np.array(image.ReadAsArray()) / rescale_value
        _, __, patch_size = image_array.shape
        image_array = image_array.transpose(1, 2, 0)
        image_array = np.expand_dims(image_array, axis=0)
        result_array = model.predict(image_array)
        if num_classes != 1:  # softmax
            result_array = np.argmax(result_array[0], axis=2)
        else:  # sigmoid
            result_array = np.around(np.reshape(result_array[0], (patch_size, patch_size)))
        outfile = output_folder + os.path.basename(test_image_paths[i])
        outdata = outdriver.Create(str(outfile), patch_size, patch_size, 1, gdal.GDT_Float64)
        outdata.GetRasterBand(1).WriteArray(result_array)
        print("Predicted " + str(i + 1) + " Images")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Name of the model, should be from unet, resunet, segnet")
    parser.add_argument("--input_shape", nargs='+', type=int, help="Input shape of the model (rows, columns, channels)")
    parser.add_argument("--weights", type=str, help="Name and path of the trained model")
    parser.add_argument("--csv_paths", type=str, help="CSV file with image and label paths")
    parser.add_argument("--output_folder", type=str, help="Folder to save the predicted images")
    parser.add_argument("--num_classes", type=int, help="Number of classes")
    parser.add_argument("--rs", type=int, help="Radiometric resolution of the image", default=8)

    args = parser.parse_args()
    predict(args.model, args.weights, tuple(args.input_shape), args.num_classes,
            args.csv_paths, args.rs, args.output_folder)
