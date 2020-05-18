import csv
import glob
import os
import argparse


def csv_gen(args):
    rows = []
    if args.pred_folder is None:
        # print(args.label_folder)
        image_paths = sorted(glob.glob(args.image_folder + "*." + args.image_format))
        label_paths = sorted(glob.glob(args.label_folder + "*." + args.label_format))
        print(len(image_paths), len(label_paths))
        for i, j in zip(image_paths, label_paths):
            # print(i, j)
            # print(os.path.splitext(os.path.basename(i))[0][:-5], os.path.splitext(os.path.basename(j))[0][:-4])
            print(os.path.splitext(os.path.basename(i))[0], os.path.splitext(os.path.basename(j))[0])
            if os.path.splitext(os.path.basename(i))[0] == os.path.splitext(os.path.basename(j))[0]:
                rows.append([i, j])
            else:
                print("Image and label names not matched")
    else:
        print(args.image_folder + "*." + args.image_format)
        # print(glob.glob(args.image_folder + "*." + args.image_format))
        image_paths = sorted(glob.glob(args.image_folder + "*." + args.image_format))
        label_paths = sorted(glob.glob(args.label_folder + "*." + args.label_format))
        pred_paths = sorted(glob.glob(args.pred_folder + "*." + args.pred_format))
        print(len(image_paths), len(label_paths), len(pred_paths))
        for i, j, k in zip(image_paths, label_paths, pred_paths):
            print(os.path.splitext(os.path.basename(i))[0], os.path.splitext(os.path.basename(j))[0], os.path.splitext(os.path.basename(k))[0])
            if os.path.splitext(os.path.basename(i))[0] == os.path.splitext(os.path.basename(j))[0] and os.path.splitext(os.path.basename(i))[0] == os.path.splitext(os.path.basename(k))[0]:
                rows.append([i, j, k])
            # print(os.path.splitext(os.path.basename(i))[0][:-4], os.path.splitext(os.path.basename(j))[0][:-5], os.path.splitext(os.path.basename(k))[0][:-4])
            # if os.path.splitext(os.path.basename(i))[0][:-4] == os.path.splitext(os.path.basename(j))[0][:-5] and os.path.splitext(os.path.basename(i))[0][:-4] == os.path.splitext(os.path.basename(k))[0][:-4]:
            #     rows.append([i, j, k])
            else:
                print("Image, label and pred names not matched")

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
    parser.add_argument("--pred_folder", type=str, help="pred_folder", default=None)
    parser.add_argument("--pred_format", type=str, help=" pred_format", default=None)
    parser.add_argument("--output_csv", type=str, help="File path of csv that gives image, label paths")
    args = parser.parse_args()
    csv_gen(args)
