import csv
import gdal
import numpy as np
import os
from tensorflow.keras.models import load_model
import argparse
from models import segnet_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def add_parser(subparser):
    parser = subparser.add_parser("visualize_test", help="Visualizing test data", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str,  help="Name of the model, should be from unet, resunet, segnet")
    parser.add_argument("--input_shape", nargs='+', type=int, help="Input shape of the model (rows, columns, channels)")
    parser.add_argument("--weights", type=str,  help="Name and path of the trained model")
    parser.add_argument("--csv_paths", type=str,  help="CSV file with image and label paths")
    parser.add_argument("--output_path", type=str,  help="Output path of pdf")
    parser.set_defaults(func=visualize_test)


def visualize_test(args):
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

    image_arrays = []
    label_arrays = []
    result_arrays = []
    # print(len(test_image_paths))
    plt.figure()
    for i in range(len(test_image_paths)):
        image = gdal.Open(test_image_paths[i])
        image_array = np.array(image.ReadAsArray())/255
        image_array1 = image_array.transpose(1, 2, 0)
        image_arrays.append(image_array1)

        label = gdal.Open(test_label_paths[i])
        label_array = np.array(label.ReadAsArray())
        label_arrays.append(label_array)

        image_array = np.expand_dims(image_array1, axis=0)
        result_array = model.predict(image_array)
        result_array = np.reshape(result_array[0], (256, 256))
        result_arrays.append(result_array)
        # print(len(image_arrays), len(label_arrays), len(result_arrays))

    # print(image_arrays[0].shape, label_arrays[0].shape, result_arrays[0].shape)
    result_arrays = np.array(result_arrays)
    print("Predicted all the images")
    # print(len(result_arrays), len(image_arrays), len(label_arrays))

    image_size = 256
    with PdfPages(args.output_path) as pdf:
        num_pages = int(len(result_arrays)/8)
        # print(num_pages)
        for i in range(num_pages):
            # print("Page number is: " + str(i))
            fig = plt.figure()
            # print(i)
            images = image_arrays[i*8:(i+1)*8]
            # print(len(images))
            labels = label_arrays[i*8:(i+1)*8]
            # print(len(labels))
            results = result_arrays[i*8:(i+1)*8]
            # print(len(results))
            columns = 6
            rows = 4
            number = 0
            for i, j, k in zip(range(1, (columns * rows) + 1, 3), range(2, (columns * rows) + 2, 3), range(3, (columns * rows) + 3, 3)):
                # print(number, i, j, k)
                fig.add_subplot(rows, columns, i)
                plt.imshow(images[number])
                plt.axis('off')
                fig.add_subplot(rows, columns, j)
                plt.imshow(labels[number])
                plt.axis('off')
                fig.add_subplot(rows, columns, k)
                plt.imshow(results[number])
                plt.axis('off')
                number += 1
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()
