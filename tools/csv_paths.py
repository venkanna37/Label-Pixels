"""
Creates CSV paths to save the images and labels location

Author: Venkanna Babu Guthula
Date: 04-07-2021
Email: g.venkanna37@gmail.com
"""

import csv
import glob
import os
import argparse


def csv_gen(args):
    rows = []
    if args.net_type == "fcn":
        image_paths = sorted(glob.glob(args.image_folder + "*." + args.image_format))
        label_paths = sorted(glob.glob(args.label_folder + "*." + args.label_format))
        print(len(image_paths), len(label_paths))
        for i, j in zip(image_paths, label_paths):
            # print(i, j)
            # print(os.path.splitext(os.path.basename(i))[0][:-5], os.path.splitext(os.path.basename(j))[0][:-4])
            # print(os.path.splitext(os.path.basename(i))[0], os.path.splitext(os.path.basename(j))[0])
            if os.path.splitext(os.path.basename(i))[0] == os.path.splitext(os.path.basename(j))[0]:
                rows.append([i, j])
            else:
                print("Image and label names not matched")
    elif args.net_type == "cnn":
        class_dirs = os.listdir(args.image_folder)  # list directories each, # write error and exceptions
        label = 0
        for i in class_dirs:
            # loop all picture in directory
            for image in glob.glob(args.image_folder + os.path.sep + i + os.path.sep + "*." + args.image_format):
                rows.append([image, label])
            label += 1

    filename = args.output_csv
    with open(filename, 'w', newline="\n") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows)
    print("CSV file created")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, help="Image folder")
    parser.add_argument("--image_format", type=str, help="Image format", default="tif")
    parser.add_argument("--label_folder", type=str, help="Label folder")
    parser.add_argument("--label_format", type=str, help="Label format", default="tif")
    parser.add_argument("--output_csv", type=str, help="CSV file name with directory")
    parser.add_argument("--net_type", type=str, help="Architecture type CNN or FCN", default="fcn")
    args = parser.parse_args()
    csv_gen(args)
