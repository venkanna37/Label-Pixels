<h1 align='center'>Label-Pixels</h1>
Label-Pixels is the tool for semantic segmentation of remote sensing imagery using Fully Convolutional Networks (FCNs).
Initially, this tool developed for extracting the road network from high-resolution remote sensing imagery. 
And now, this tool can be used to extract various features (Semantic segmentation of remote sensing imagery). 
This is part of my MSc research work (Automatic Road Extraction from High-Resolution Remote Sensing Imagery 
using Fully Convolutional Networks and Transfer Learning), worked under Mr. Ashutosh Kumar Jha 
(IIRS) and Dr. Claudio Persello (University of Twente, ITC).

### Tested on
```
python 3.6.13
Keras 2.3.1
Tensorflow 2.1.0
gdal 2.2.2
scikit-learn 0.23.2
```
### Installation
#### Clone repository
```commandline
git clone https://github.com/venkanna37/Label-Pixels.git
cd Label-Pixels
```
#### Setup environment
* Create an environment and install above packages OR install with `environment.yml` file in Anaconda
* Note: `environment.yml` file created with `conda env export > environment.yml` command in Anaconda
```commandline
conda env create -f environment.yml
conda activate label-pixels
cd tools
```
### Usage

<p align="center">
  <img width="900" height="370"  src="/data/methodology2.png">
</p>

###  Rasterize
* Creates labels with shapefiles
* The projection of imagery and shapefiles should be same
* Projection units should be in meters if you want to buffer line feature
* `--buffer` not required for polygon
* `--labels_atr` not required for line and single class
* output directory has to create to save label images `Ex: ../data/spacenet/labels/`

| options | Description |
----------|--------------
--help| Print usage information
--raster_dir| Directory that contains raster image/images
--vector_dir| Directory that contains vector files with the same projection as raster data. And name of the vector and raster files should be same.
--raster_format| Raster format of the image/images
--vector_format| Vector format ex: shp, geojson, etc.
--output_dir| Output directory to save labels
--buffer| Buffer length for line feature. Not required for polygon
--buffer_atr| Attribute from the vector file, this attribute can be buffer width and it multiplies with `--buffer`.
--labels_atr| Attribute from the vector file, pixels inside the polygon will be assigned by its attribute value. 

<b>Examples:</b>

python rasterize.py --raster_dir ../data/spacenet/raster/ --raster_format tif
 --vector_dir ../data/spacenet/vector/ --vector_format shp --buffer 2
  --output_dir ../data/spacenet/labels/ --label_atr partialDec --buffer_atr lanes

python rasterize.py --raster_dir ../data/spacenet/raster/ --vector_dir ../data/spacenet/vector_multi/
 --vector_format shp --output_dir ../data/spacenet/labels/ --label_atr value

###  Patch Generation
* Generate patches from images/tiles
* To generate patches for train, test and validation sets, the command needs to be run three times
* Name of image and label files should be same
* Output directory has to create to save patches `Ex: ../data/mass_patches/`

| options | Description |
----------|--------------
--image_folder | Folder of input images/tiles with directory
--image_format | Image format tiff/tif/jpg/png
--label_folder | Folder of label images with directory
--label_format | Label format tiff/tif/jpg/png
--patch_size | Patch size to feed network. Default size is 256
--overlap | Overlap between two patches on image/tile (units: pixels)
--output_folder | Output folder to save patches

<b> Example: </b>

python patch_gen.py --image_folder ../data/mass_sample/test/image/ --image_format tiff
 --label_folder ../data/mass_sample/test/roads_and_buildings/ --label_format tif --patch_size 256
 --output_folder ../data/mass_patches/

### CSV Paths
* Save directories of patches in CSV file instead of reading patches from folders directly
* Output directory has to create to save the csv files `Ex: ../paths/`

| options | Description |
----------|--------------
--image_folder | Folder of image patches with directory
--image_format | Image format tif (patch_gen.py save patches in tif format)
--label_folder | Folder of label patches with directory
--label_format | Label format tif (patch_gen.py save patches in tif format)
--patch_size | Patch size to feed network. Default size is 256
--output_csv | csv filename with directory

<b> Example </b>

python csv_paths.py --image_folder ../data/mass_patches/image/ --label_folder ../data/mass_patches/label/
 --output_csv ../paths/data_rnb.csv

###  Training
* Training FCNs (UNet, SegNet, ResUNet and UNet-Mini) for semantic segmentation 
* For Binary classification, `--num_classes = 1`
* For Binary classification with one-hot encoding, `--num_classes = 2`
* For multi class classification, `--num_classes = number of target classes (>1)`

| options | Description |
----------|--------------
--model | Name of the FCN model. Existing models are unet, unet_mini, segnet and resunet
--train_csv | CSV file name with directory, consists of directories of image and label patches of training set.
--valid_csv | CSV file name with directory, consists of directories of image and label patches of validation set.
--input_shape | Input shape of model to feed patches (patch_size patch_size channels)
--batch_size | Batch size, depends on GPU/CPU memory
--num_classes | Number of classes in labels data
--epochs | Number of epochs

<b> Example </b>

python train.py --model unet_mini --train_csv ../paths/data_rnb.csv --valid_csv ../paths/data_rnb.csv
--input_shape 256 256 3 --batch_size 4 --num_classes 3 --epochs 100

###  Accuracy
* Calculates the accuracy using different accuracy metrics.
* IoU, F1-Score, Precision and Recall

| options | Description |
----------|--------------
--input_shape | Input shape of model (patch_size, patch_size, channels)
--weights | Trained model with directory
--csv_paths | CSV file name with directory, consists of directories of image and label patches of test set.
--num_classes | Number of classes in labels data

<b> Example </b>

python accuracy.py --model unet_mini --input_shape 256 256 3 --weights ../trained_models/unet_mini300_06_07_20.hdf5
--csv_paths ../paths/data_rnb.csv --num_classes 3

###  Prediction
* Predicts the entire image/tile with trained model.
* Output directory has to create to save the predicted images `Ex: ../data/predictions/`

| options | Description |
----------|--------------
--input_shape | Input shape of model (patch_size, patch_size, channels)
--weights | Trained model with directory
--image_folder | Folder of input images/tiles with directory
--image_format | Image format tiff/tif/jpg/png
--output_folder | Output folder to save predicted images/tiles

<b> Example: </b>

python tile_predict.py --model unet_mini --input_shape 256 256 3 --weights ../trained_models/unet_mini300_06_07_20.hdf5
--image_folder ../data/mass_sample/test/image/ --image_format tiff --output_folder ../data/predictions/ --num_classes 3

### Summary of the Model
* Summary of FCNs
* Useful to check the configuration of Fully Convolutional Networks
* Replace `unet, segnet and resunet` with `unet_mini` to check configuration of all netowrks

| options | Description |
----------|--------------
--model | Name of FCN model. Existing models are unet, unet_mini, segnet and resunet
--input_shape | Input shape of model to feed (patch_size patch_size channels)
--num_classes | Number of classes to train

<b> Example </b>

python summary.py --model unet_mini --input_shape 256 256 3 --num_classes 3

### Example Outputs
<p align="center">
  <img width="900" height="1300"  src="/data/mass_sota.png">
  <img width="900" height="330"  src="/data/mass_roads_and_buildings.png">
</p>

### Benchmark datasets
1. Massachusetts Benchmark datasets for Roads and Buildings extraction <br/>
[https://academictorrents.com/browse.php?search=Volodymyr
   +Mnih](https://academictorrents.com/browse.php?search=Volodymyr+Mnih)
2. List of Benchmark datasets for semantic segmentation, object detection from remote sensing imagery
[https://github.com/chrieke/awesome-satellite-imagery
   -datasets](https://github.com/chrieke/awesome-satellite-imagery-datasets)

#### Any problem with code?
Open [issue](https://github.com/venkanna37/Label-Pixels/issues) or mail me :point_right:  g.venkanna37@gmail.com
