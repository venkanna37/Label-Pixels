"""
Creatig patches from image/list of images

Author: Venkanna Babu Guthula
Date: 03-07-2021
Email: g.venkanna37@gmail.com
"""

import gdal
import glob
import numpy as np
import os
import argparse
from models import lp_utils as lu


def patch_gen(x):
    # Reading all the images and labels from input folders
    image_paths = sorted(glob.glob(str(x.image_folder) + "*." + x.image_format))
    label_paths = sorted(glob.glob(str(x.label_folder) + "*." + x.label_format))
    # Creating output folders for patch size images
    os.mkdir(x.output_folder + "image/")
    os.mkdir(x.output_folder + "label/")
    output_image_folder = x.output_folder + "image/"
    output_label_folder = x.output_folder + "label/"

    outdriver = gdal.GetDriverByName("GTiff")
    # Clipping images with patch_size
    format_len_i = len(x.image_format)+1
    format_len_l = len(x.label_format)+1
    for h in range(len(image_paths)):
        image_path = image_paths[h]
        label_path = label_paths[h]
        _, image_name = os.path.split(image_path)
        image_name = image_name[:-format_len_i-x.slice_im]
        __, label_name = os.path.split(label_path)
        label_name = label_name[:-format_len_l-x.slice_la]
        print(image_name, label_name)
        if image_name == label_name:
            image = gdal.Open(image_path)
            image_array = np.array(image.ReadAsArray())
            label = gdal.Open(label_path)
            if x.dataset == "deep_lulc":
                label_array = np.array(label.ReadAsArray())
                num_channels, num_rows, num_cols = image_array.shape
            else:
                label_array = np.array(label.GetRasterBand(1).ReadAsArray())
                label_array = np.expand_dims(label_array, axis=0)
                num_channels, num_rows, num_cols = image_array.shape

            for i in range(int(num_rows / x.patch_size)):
                for j in range(int(num_cols / x.patch_size)):
                    x1 = (i * x.patch_size) - (i * x.overlap)
                    y1 = ((i + 1) * x.patch_size) - (i * x.overlap)
                    x2 = (j * x.patch_size) - (j * x.overlap)
                    y2 = ((j + 1) * x.patch_size) - (j * x.overlap)
                    temp_image = image_array[:, x1:y1, x2:y2]
                    temp_label = label_array[:, x1:y1, x2:y2]
                    outfile_label = output_label_folder + label_name + '_' + str(i) + '_' + str(j) + '.tif'
                    outfile_image = output_image_folder + image_name + '_' + str(i) + '_' + str(j) + '.tif'
                    outdata_image = outdriver.Create(str(outfile_image), x.patch_size, x.patch_size, 3)
                    outdata_label = outdriver.Create(str(outfile_label), x.patch_size, x.patch_size, 1)
                    if x.dataset == "deep_lulc":
                        temp_label = lu.deep_lulc_data(temp_label)
                    else:
                        temp_label = temp_label[0]
                    outdata_label.GetRasterBand(1).WriteArray(temp_label)
                    for k in range(len(temp_image)):
                        outdata_image.GetRasterBand(k + 1).WriteArray(temp_image[k])
            print('Clipped ' + str(h) + ' images')
        else:
            print("Image name and label names did not match, images and labels names are expected with same names")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, help="Folder of input images")
    parser.add_argument("--image_format", type=str, help="Image format", default="tif")
    parser.add_argument("--label_folder", type=str, help="Folder of corresponding labels")
    parser.add_argument("--label_format", type=str, help="Label format", default="tif")
    parser.add_argument("--patch_size", type=int, help="Patch size to split the tiles/images", default=256)
    parser.add_argument("--overlap", type=int, help="Overlap between two patches", default=0)
    parser.add_argument("--slice_im", type=int, help="Slicing image name", default=0)
    parser.add_argument("--slice_la", type=int, help="Slicing label name", default=0)
    parser.add_argument("--output_folder", type=str, help="Output folder to save images and labels")
    parser.add_argument("--dataset", type=str, help="label preparation for custom data", default="venky")
    args = parser.parse_args()
    patch_gen(args)
