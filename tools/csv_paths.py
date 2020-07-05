import csv
import glob
import os
import argparse


def csv_gen(args):
    rows = []
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

    filename = args.output_csv
    with open(filename, 'w') as csvfile:
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
    args = parser.parse_args()
    csv_gen(args)
