import gdal
import csv
import numpy as np
from tensorflow.keras.models import load_model
from models import segnet_model
import os
from sklearn.metrics import confusion_matrix
import argparse
import sys

# np.set_printoptions(threshold=sys.maxsize)


def gtiff_to_array(file_path):
    """Takes a file path and returns a tif file as a 3-dimensional numpy array, width x height x bands."""
    data = gdal.Open(file_path)
    bands = [data.GetRasterBand(i + 1).ReadAsArray() for i in range(data.RasterCount)]
    x = np.stack(bands, axis=2)
    return np.stack(bands, axis=2)


def gridwise_sample(imgarray, patchsize, tilesize):
    """Extract sample patches of size patchsize x patchsize from an image (imgarray) in a gridwise manner.
    """
    patchidx = []
    nrows, ncols, nbands = imgarray.shape
    patchsamples = np.zeros(shape=(0, patchsize, patchsize, nbands),
                            dtype=imgarray.dtype)
    for i in range(int(tilesize / patchsize)):
        for j in range(int(tilesize / patchsize)):
            tocat = imgarray[i * patchsize:(i + 1) * patchsize,
                    j * patchsize:(j + 1) * patchsize, :]
            tocat = np.expand_dims(tocat, axis=0)/255
            patchsamples = np.concatenate((patchsamples, tocat), axis=0)
            patchidx.append([i, j])
    return patchsamples, patchidx


def image_from_patches(result, idx, patchsize, tilesize):
    outimage = np.zeros((tilesize, tilesize, 1))
    if len(result) == len(idx):
        for k in range(len(idx)):
            i, j = idx[k][0], idx[k][1]
            outimage[i * patchsize: (i + 1) * patchsize, j * patchsize: (j + 1) * patchsize, :] = result[k]
    return outimage


def gridwise_sample_mass(imgarray, patchsize, tilesize):
    """Extract sample patches of size patchsize x patchsize from an image (imgarray) in a gridwise manner.
    """
    patchidx = []
    nrows, ncols, nbands = imgarray.shape
    patchsamples = np.zeros(shape=(0, patchsize, patchsize, nbands),
                            dtype=imgarray.dtype)
    for i in range(round(tilesize / patchsize)):
        for j in range(round(tilesize / patchsize)):
            if i < 5 and j < 5:
                tocat = imgarray[i * patchsize:(i + 1) * patchsize, j * patchsize:(j + 1) * patchsize, :]
                tocat = np.expand_dims(tocat, axis=0)/255
                patchsamples = np.concatenate((patchsamples, tocat), axis=0)
                patchidx.append([i, j])

            elif j == 5 and i < 5:
                tocat = imgarray[i * patchsize:(i + 1) * patchsize, 1244:, :]
                tocat = np.expand_dims(tocat, axis=0)/255
                patchsamples = np.concatenate((patchsamples, tocat), axis=0)
                patchidx.append([i, j])

            elif i == 5 and j < 5:
                tocat = imgarray[1244:, j * patchsize:(j + 1) * patchsize, :]
                tocat = np.expand_dims(tocat, axis=0)/255
                patchsamples = np.concatenate((patchsamples, tocat), axis=0)
                patchidx.append([i, j])

            elif i == 5 and j == 5:
                tocat = imgarray[1244:, 1244:, :]
                tocat = np.expand_dims(tocat, axis=0)/255
                patchsamples = np.concatenate((patchsamples, tocat), axis=0)
                patchidx.append([i, j])
    return patchsamples, patchidx


def image_from_patches_mass(result, idx, patchsize, tilesize):
    outimage = np.zeros((tilesize, tilesize, 1))
    if len(result) == len(idx):
        for k in range(len(idx)):
            i, j = idx[k][0], idx[k][1]
            if i < 5 and j < 5:
                outimage[i * patchsize:(i + 1) * patchsize, j * patchsize:(j + 1) * patchsize, :] = result[k]
            elif j == 5 and i < 5:
                outimage[i * patchsize:(i + 1) * patchsize, 1244:, :] = result[k]
            elif i == 5 and j < 5:
                outimage[1244:, j * patchsize:(j + 1) * patchsize, :] = result[k]
            elif i == 5 and j == 5:
                outimage[1244:, 1244:, :] = result[k]
    return outimage


def tile_predict(args):
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
    # rows = []
    for i in range(len(test_image_paths)):
        image_arrays = gtiff_to_array(test_image_paths[i])
        if args.tile_size == 1500:
            image_patches, patchidx = gridwise_sample_mass(image_arrays, args.patch_size, args.tile_size)
        elif args.tile_size == 1024:
            image_patches, patchidx = gridwise_sample(image_arrays, args.patch_size, args.tile_size)
        else:
            print("Tile size either 1500 or 1025")
        result = model.predict(image_patches)
        if args.tile_size == 1500:
            result_array = image_from_patches_mass(result, patchidx, args.patch_size, args.tile_size)
        elif args.tile_size == 1024:
            result_array = image_from_patches(result, patchidx, args.patch_size, args.tile_size)
        # result_array = np.reshape(result_array, (args.tile_size, args.tile_size))  #  Value between 0 and 1
        result_array = np.around(np.reshape(result_array, (args.tile_size, args.tile_size)))  # Binary, 0 and 1
        filename = os.path.splitext(os.path.basename(test_image_paths[i]))[0]
        outfile = args.output_folder + filename
        # print(filename)
        outfile = outfile + ".tif"  # Line for jpg, png, and tiff formats
        outdata = outdriver.Create(str(outfile), args.tile_size, args.tile_size, 1, gdal.GDT_Int16)
        outdata.GetRasterBand(1).WriteArray(result_array)
        print("Predicted " + str(i + 1) + " Images")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Name of the model, should be from unet, resunet, segnet")
    parser.add_argument("--input_shape", nargs='+', type=int, help="Input shape of the model (rows, columns, channels)")
    parser.add_argument("--weights", type=str, help="Name and path of the trained model")
    parser.add_argument("--csv_paths", type=str, help="CSV file with image and label paths")
    parser.add_argument("--patch_size", type=int, help="Patch size of the tile")
    parser.add_argument("--tile_size", type=int, help="Images size expected (Tile should be in square size")
    parser.add_argument("--output_folder", type=str, help="Output path of the predicted images")
    args = parser.parse_args()
    tile_predict(args)
