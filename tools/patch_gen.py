"""
Creating patches from image/list of images
:keyword
patch: size of an image that feed to deep learning models
"""

import gdal
import glob
import numpy as np
import os
import shutil
import argparse


def patch_gen(image_folder, image_format, label_folder, label_format,
              patch_size, overlap, output_folder):
    # Reading all the images and labels from input folders
    image_paths = sorted(glob.glob(str(image_folder) + "*." + image_format))
    label_paths = sorted(glob.glob(str(label_folder) + "*." + label_format))

    # Creating output folders for patch size images
    output_image_folder = output_folder + "image/"
    output_label_folder = output_folder + "label/"
    if os.path.exists(output_image_folder):
        shutil.rmtree(output_image_folder)
    if os.path.exists(output_label_folder):
        shutil.rmtree(output_label_folder)
    os.mkdir(output_image_folder)
    os.mkdir(output_label_folder)


    outdriver = gdal.GetDriverByName("GTiff")
    # Clipping images with patch_size
    format_len_i = len(image_format)+1
    format_len_l = len(label_format)+1
    for h in range(len(image_paths)):
        image_path = image_paths[h]
        label_path = label_paths[h]
        _, image_name = os.path.split(image_path)
        image_name = image_name[:-format_len_i]
        __, label_name = os.path.split(label_path)
        label_name = label_name[:-format_len_l]
        print(image_name, label_name)
        if image_name == label_name:
            image = gdal.Open(image_path)
            image_array = np.array(image.ReadAsArray())
            label = gdal.Open(label_path)
            label_array = np.array(label.GetRasterBand(1).ReadAsArray())
            label_array = np.expand_dims(label_array, axis=0)
            if len(image_array.shape) == 2:
                num_channels = 1
                num_rows, num_cols = image_array.shape
            else:
                num_channels, num_rows, num_cols = image_array.shape

            for i in range(int(num_rows / patch_size)):
                for j in range(int(num_cols / patch_size)):
                    x1 = (i * patch_size) - (i * overlap)
                    y1 = ((i + 1) * patch_size) - (i * overlap)
                    x2 = (j * patch_size) - (j * overlap)
                    y2 = ((j + 1) * patch_size) - (j * overlap)
                    temp_image = image_array[:, x1:y1, x2:y2]
                    temp_label = label_array[:, x1:y1, x2:y2]
                    outfile_label = output_label_folder + label_name + '_' + str(i) + '_' + str(j) + '.tif'
                    outfile_image = output_image_folder + image_name + '_' + str(i) + '_' + str(j) + '.tif'
                    outdata_image = outdriver.Create(str(outfile_image), patch_size, patch_size, num_channels)
                    outdata_label = outdriver.Create(str(outfile_label), patch_size, patch_size, 1)
                    outdata_label.GetRasterBand(1).WriteArray(temp_label[0])
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
    parser.add_argument("--output_folder", type=str, help="Output folder to save images and labels")
    args = parser.parse_args()

    patch_gen(args.image_folder, args.image_format, args.label_folder, args.label_format,
              args.patch_size, args.overlap, args.output_folder)
