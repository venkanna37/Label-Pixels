import gdal
import csv
import numpy as np
from tensorflow.keras.models import load_model
from models import segnet_model
import os
from sklearn.metrics import confusion_matrix
import argparse
import sys
import glob


def gtiff_to_array(file_path):
    """Takes a file path and returns a tif file as a 3-dimensional numpy array, width x height x bands."""
    data = gdal.Open(file_path)
    bands = [data.GetRasterBand(i + 1).ReadAsArray() for i in range(data.RasterCount)]
    x = np.stack(bands, axis=2)
    return np.stack(bands, axis=2)


def gridwise_sample(imgarray, patchsize):
    """Extract sample patches of size patchsize x patchsize from an image (imgarray) in a gridwise manner.
    """
    patchidx = []
    nrows, ncols, nbands = imgarray.shape
    patchsamples = np.zeros(shape=(0, patchsize, patchsize, nbands),
                            dtype=imgarray.dtype)
    for i in range(round(nrows / patchsize)):
        for j in range(round(ncols / patchsize)):
            if i+1 <= int(nrows / patchsize) and j+1 <= int(ncols / patchsize):
                tocat = imgarray[i * patchsize:(i + 1) * patchsize,
                        j * patchsize:(j + 1) * patchsize, :]
                tocat = np.expand_dims(tocat, axis=0) / 255
                patchsamples = np.concatenate((patchsamples, tocat), axis=0)
                patchidx.append([i, j])
            elif i+1 <= int(nrows / patchsize) and j+1 >= int(ncols / patchsize):
                tocat = imgarray[i * patchsize:(i + 1) * patchsize, -patchsize:, :]
                tocat = np.expand_dims(tocat, axis=0) / 255
                patchsamples = np.concatenate((patchsamples, tocat), axis=0)
                patchidx.append([i, j])
            elif i+1 >= int(nrows / patchsize) and j+1 <= int(ncols / patchsize):
                tocat = imgarray[-patchsize:, j * patchsize:(j + 1) * patchsize, :]
                tocat = np.expand_dims(tocat, axis=0) / 255
                patchsamples = np.concatenate((patchsamples, tocat), axis=0)
                patchidx.append([i, j])
            elif i+1 >= int(nrows / patchsize) and j+1 >= int(ncols / patchsize):
                tocat = imgarray[-patchsize:, -patchsize:, :]
                tocat = np.expand_dims(tocat, axis=0) / 255
                patchsamples = np.concatenate((patchsamples, tocat), axis=0)
                patchidx.append([i, j])

    return patchsamples, patchidx


def image_from_patches(result, idx, patchsize, t_rows, t_cols):
    outimage = np.zeros((t_rows, t_cols, 1))
    if len(result) == len(idx):
        for k in range(len(idx)):
            i, j = idx[k][0], idx[k][1]
            if i+1 <= int(t_rows / patchsize) and j+1 <= int(t_cols / patchsize):
                outimage[i * patchsize:(i + 1) * patchsize, j * patchsize:(j + 1) * patchsize, :] = result[k]
            elif i+1 <= int(t_rows / patchsize) and j + 1 >= int(t_cols / patchsize):
                outimage[i * patchsize:(i + 1) * patchsize, -patchsize:, :] = result[k]
            elif i+1 >= int(t_rows / patchsize) and j + 1 <= int(t_cols / patchsize):
                outimage[-patchsize:, j * patchsize:(j + 1) * patchsize, :] = result[k]
            elif i+1 >= int(t_rows / patchsize) and j + 1 >= int(t_cols / patchsize):
                outimage[-patchsize:, -patchsize:, :] = result[k]
    return outimage

def tile_predict(args):
    if args.model == "unet":
        model = load_model(args.weights)
    elif args.model == "resunet":
        model = load_model(args.weights)
    elif args.model == "segnet":
        model = segnet_model.create_segnet(args)
        model.load_weights(args.weights)

    _, p_rows, p_cols, p_chan = model.layers[0].input_shape[0] # Patch shape
    image_paths = sorted(glob.glob(args.image_folder + "*." + args.image_format))
    outdriver = gdal.GetDriverByName("GTiff")

    for i in range(len(image_paths)):
        image = gdal.Open(image_paths[i])
        image_array = np.array(image.ReadAsArray())
        image_array = image_array.transpose(1, 2, 0)
        t_rows, t_cols, t_chan = image_array.shape
        image_patches, patchidx = gridwise_sample(image_array, p_rows) #Tile size takes from image and patch size from trained model
        result = np.zeros(shape=(0, p_rows, p_cols, 1))
        for j in range(image_patches.shape[0]):
            patch = np.expand_dims(image_patches[j], axis=0)
            patch_result = model.predict(patch)
            patch_result = np.expand_dims(np.argmax(patch_result, axis=3), axis=-1)
            result = np.concatenate((result, patch_result), axis=0)
        # print(result.shape)
        result_array = image_from_patches(result, patchidx, p_rows, t_rows, t_cols)
        # result_array = np.reshape(result_array, (args.tile_size, args.tile_size))  #  Value between 0 and 1
        result_array = np.reshape(result_array, (t_rows, t_cols))  # Binary, 0 and 1
        filename = os.path.splitext(os.path.basename(image_paths[i]))[0]
        outfile = args.output_folder + filename
        outfile = outfile + ".tif"  # Line for jpg, png, and tiff formats
        outdata = outdriver.Create(str(outfile), t_rows, t_cols, 1, gdal.GDT_Int16)
        outdata.GetRasterBand(1).WriteArray(result_array)
        if image.GetProjection() and image.GetGeoTransform():
            proj = image.GetProjection()
            trans = image.GetGeoTransform()
            outdata.SetProjection(proj)
            outdata.SetGeoTransform(trans)
        print("Predicted " + str(i + 1) + " Images")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Name of the model, should be from unet, resunet, segnet")
    parser.add_argument("--input_shape", nargs='+', type=int, help="Input shape of the model (rows, columns, channels)")
    parser.add_argument("--weights", type=str, help="Name and path of the trained model")
    parser.add_argument("--image_folder", type=str, help="Folder of image or images")
    parser.add_argument("--image_format", type=str, help="Image format")
    parser.add_argument("--output_folder", type=str, help="Output path of the predicted images")
    args = parser.parse_args()
    tile_predict(args)
