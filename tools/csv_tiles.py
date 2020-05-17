import csv
import glob
import os
import argparse


def add_parser(subparser):
    parser = subparser.add_parser("csv_tiles", help="Generating csv paths", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_folder", type=str,  help="Image folder")
    parser.add_argument("--image_format", type=str,  help="Image format")
    parser.add_argument("--label_folder", type=str,  help="Label folder")
    parser.add_argument("--label_format", type=str,  help="Label format")
    parser.add_argument("--output_csv", type=str,  help="File path of csv that gives image, label paths")
    parser.set_defaults(func=csv_gen2)


def csv_gen2(args):
    image_paths = sorted(glob.glob(args.image_folder + "*." + args.image_format))
    label_paths = sorted(glob.glob(args.label_folder + "*." + args.label_format))
    print(len(image_paths), len(label_paths))
    rows = []
    for h in range(len(image_paths)):
        image_path = image_paths[h]
        _, image_name = os.path.split(image_path)
        image_name = image_name[:-5]
        for i in range(len(label_paths)):
            label_path = label_paths[i]
            __, label_name = os.path.split(label_path)
            label_name = label_name[:-4]
            if image_name == label_name:
                rows.append([image_paths[h], label_paths[i]])

    filename = args.output_csv
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        # csvwriter.writerow(fields)
        csvwriter.writerows(rows)
