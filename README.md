<h1 align='center'>Label-Pixels</h1>
Label-Pixels is a tool for semantic segmentation of remote sensing images using fully convolutional networks (FCNs), designed for extracting the road network from remote sensing imagery and it can be
 used in other applications applications to label every pixel in the image ( Semantic segmentation).
  This is part of my MSc research project (Automatic Road Extraction from High-Resolution Remote Sensing Imagery
  using Fully Convolutional Networks and Transfer Learning).


## Clone repository and install packages in Anaconda
```commandline
git clone https://github.com/venkanna37/Label-Pixels.git
```
```commandline
conda env create -f environment.yml
```
OR
```commandline
conda install -c conda-forge keras
conda install -c conda-forge gdal
conda install -c anaconda scikit-learn
```
## Usage

<p align="center">
  <img width="900" height="370"  src="/data/methodology2.png">
</p>

##  Rasterize
* Creates labels with shapefiles
* The projection of imagery and shapefiles should be same
* Projection units should be in meters if you want to buffer line feature

| options | Description |
----------|--------------
--help| Print usage information
--raster| Raster/Image name with directory
--vector| Vector file name with directory
--output_file| Output filename with directory
--buffer| Buffer length for line feature. Not required for polygon
--buffer_atr| Attribute from the vector file, this attribute can be buffer width and It multiplies with --buffer. Not required for polygon
--labels_atr| Attribute from the vector file, pixels inside the polygon will be assigned by its attribute value. Not required for line

<b>Example:</b>

python rasterize.py --raster ../data/spacenet/raster/spacenet_chip0.tif --vector ../data/spacenet/vector/spacenet_chip0.shp --buffer 2 --buffer_atr lanes --output_file ../data/spacenet/binary/test.tif

##  Patch Generation
* Generate patches from Images/Tiles
* To generate patches for train, test and valid sets, the command needs to be run three times
* Name of image and label files should be same

| options | Description |
----------|--------------
--image_folder | Folder of input images/tiles with directory
--image_format | Image format tiff/tif/jpg/png
--label_folder | Folder of label images with directory
--label_format | Label format tiff/tif/jpg/png
--patch_size | Patch size to feed network. Default size is 256
--overlap | Overlap between two patches on image/tile
--output_folder | Output folder to save patches

<b> Example: </b>

python patch_gen.py --image_folder ../data/mass_sample/test/image/ --image_format tiff --label_folder ../data/mass_sample/test/roads_and_buildings/ --label_format tif --patch_size 256 --output_folder ../data/mass_patches/

## CSV Paths
* Save directories of patches in CSV file instead of reading patches from folders directly

| options | Description |
----------|--------------
--image_folder | Folder of image patches with directory
--image_format | Image format tif (patch_gen.py save patches in tif format)
--label_folder | Folder of label patches with directory
--label_format | Label format tif (patch_gen.py save patches in tif format)
--patch_size | Patch size to feed network. Default size is 256
--output_csv | csv filename with directory

<b> Example </b>

python csv_paths.py --image_folder ../data/mass_patches/image/ --image_format tif --label_folder ../data/mass_patches/label/ --label_format tif --output_csv ../paths/data_rd.csv


##  Training
* Training FCNs for semantic segmentation

| options | Description |
----------|--------------
--model | Name of FCN model. Existing models are unet, segnet and resunet
--train_csv | CSV file name with directory, consists of directories of image and label patches of training set.
--valid_csv | CSV file name with directory, consists of directories of image and label patches of validation set.
--input_shape | Input shape of model to feed (patch_size patch_size channels)
--batch_size | Batch size, depends on GPU/CPU memory
--num_classes | Number of classes in labels data
--epochs | Number of epochs

<b> Example </b>

python train.py --model unet --train_csv ../paths/data_rd.csv --valid_csv ../paths/data_rd.csv --input_shape 256 256 3 --batch_size 1 --num_classes 3 --epochs 100

##  Accuracy
* Calculates the accuracy using different accuracy metrics.

| options | Description |
----------|--------------
--input_shape | Input shape of model (patch_size patch_size channels)
--weights | Trained model with directory
--csv_paths | CSV file name with directory, consists of directories of image and label patches of test set.
--num_classes | Number of classes in labels data

<b> Example </b>

python accuracy.py --model unet --input_shape 256 256 3 --weights ../trained_models/unet300_06_07_20.hdf5 --csv_paths ../paths/data_rd.csv --num_classes 3

##  Prediction
* Predicts the the entire image/tile with trained model.

| options | Description |
----------|--------------
--input_shape | Input shape of model (patch_size patch_size channels)
--weights | Trained model with directory
--image_folder | Folder of input images/tiles with directory
--image_format | Image format tiff/tif/jpg/png
--output_folder | Output folder to save predicted images/tiles

<b> Example: </b>

python tile_predict.py --model unet --input_shape 256 256 3 --weights ../trained_models/unet300_06_07_20.hdf5 --image_folder ../data/mass_sample/test/image/ --image_format tiff --output_folder ../data/

## Summary of the Model
* Summary of FCNs

| options | Description |
----------|--------------
--model | Name of FCN model. Existing models are unet, segnet and resunet
--input_shape | Input shape of model to feed (patch_size patch_size channels)
--num_classes | Number of classes to train

<b> Example </b>

python summary.py --model unet --input_shape 256 256 3 --num_classes 3

## Example Outputs
<p align="center">
  <img width="900" height="1300"  src="/data/mass_sota.png">
  <img width="900" height="330"  src="/data/mass_roads_and_buildings.png">
</p>

## Benchmark datasets
1. Massachusetts Benchmark datasets for Roads and Buildings extraction <br/>
[https://academictorrents.com/browse.php?search=Volodymyr+Mnih](https://academictorrents.com/browse.php?search=Volodymyr+Mnih)
2. List of Benchmark datasets for semantic segmentation, object detection from remote sensing imagery
[https://github.com/chrieke/awesome-satellite-imagery-datasets](https://github.com/chrieke/awesome-satellite-imagery-datasets)

### Any problem with code?
Open [issue](https://github.com/venkanna37/Label-Pixels/issues) or mail me :point_right:  g.venkanna37@gmail.com
