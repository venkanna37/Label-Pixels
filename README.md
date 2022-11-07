<h1 align='center'>Label-Pixels</h1>

Label-Pixels is the tool for semantic segmentation of remote sensing imagery using Fully Convolutional Networks (FCNs).
Initially, this tool developed for road extraction from high-resolution remote sensing imagery and now, this tool can be
used to extract various features. This is part of my MSc research work (Automatic Road Extraction from High-Resolution
Remote Sensing Imagery using Fully Convolutional Networks and Transfer Learning), worked under Mr. Ashutosh Kumar Jha 
[(IIRS)](https://www.iirs.gov.in/) and Dr. Claudio Persello [(ITC, University of Twente)](https://www.itc.nl/).

### Tested on
```
python 3.7.15
Keras 2.9.0 
Tensorflow 2.9.2
gdal 2.2.3
scikit-learn 1.0.2
```

### Checkout Demo in Google Colab
:point_right: [Google Colab Notebook](https://colab.research.google.com/drive/1vPDrWKzwqIelCkZKi_GfXFdFGh4RHNL6?usp=sharing)

### Installation
#### Clone repository
```commandline
git clone https://github.com/venkanna37/Label-Pixels.git
cd Label-Pixels
```

#### Setup environment
* Create an environment and install above packages OR install with `environment.yml`
file in Anaconda
* Note: `environment.yml` file created with `conda env export > environment.yml`
command in Anaconda (this file will clean and updated soon)
```commandline
conda env create -f environment.yml
conda activate label-pixels
cd tools
```
### Usage

<p align="center">
  <img width="900" height="370"  src="/data/figures/methodology.png">
</p>

###  Rasterize
* Create labels with shapefiles
* The projection of imagery and shapefiles should be same
* Projection units should be in meters if you want to buffer line feature
* `--buffer` not required for polygon
* `--labels_atr` not required for line and single class
* Output directory has to create to save label images `Eg: ../data/spacenet/labels/`

| options         | Description                                                                                                                           |
|-----------------|---------------------------------------------------------------------------------------------------------------------------------------|
| --help          | Print usage information                                                                                                               |
| --raster_dir    | Directory that contains raster image/images                                                                                           |
| --vector_dir    | Directory that contains vector files with the same projection as raster data. And name of the vector and raster files should be same. |
| --raster_format | Raster format of the image/images                                                                                                     |
| --vector_format | Vector format ex: shp, geojson, etc.                                                                                                  |
| --output_dir    | Output directory to save labels                                                                                                       |
| --buffer        | Buffer length for line feature. Not required for polygon                                                                              |
| --buffer_atr    | Attribute from the vector file, this attribute can be buffer width and it multiplies with `--buffer`.                                 |
| --labels_atr    | Attribute from the vector file, pixels inside the polygon will be assigned by its attribute value.                                    |

<b>Examples:</b>

python rasterize.py --raster_dir ../data/spacenet/raster/ --raster_format tif
--vector_dir ../data/spacenet/vector/ --vector_format shp --buffer 2 --output_dir ../data/spacenet/labels/
--label_atr partialDec --buffer_atr lanes

python rasterize.py --raster_dir ../data/spacenet/raster/ --vector_dir ../data/spacenet/vector_multi/
 --vector_format shp --output_dir ../data/spacenet/labels/ --label_atr value

###  Patch Generation
* Generate patches (small image subset with fixed size) from images/tiles
* To generate patches for train, test and validation sets, the command needs to be run three times
or can create patches with single run and split dataset with `split_dataset.py` script
* Name of input and label images should be same
* Output directory has to create to save patches `Ex: ../data/mass_patches/`

| options         | Description                                               |
|-----------------|-----------------------------------------------------------|
| --image_folder  | Folder of input images/tiles with directory               |
| --image_format  | Image format tiff/tif/jpg/png                             |
| --label_folder  | Folder of label images with directory                     |
| --label_format  | Label format tiff/tif/jpg/png                             |
| --patch_size    | Patch size to feed network. Default size is 256           |
| --overlap       | Overlap between two patches on image/tile (units: pixels) |
| --output_folder | Output folder to save patches                             |

<b> Example: </b>

python patch_gen.py --image_folder ../data/massachusetts/test/image/ --image_format tiff
--label_folder ../data/massachusetts/test/roads_and_buildings/ --label_format tif
--patch_size 256 --output_folder ../data/mass_patches/

### CSV Paths
* Save directories of patches in CSV file instead of reading patches from folders directly
* Output directory has to create to save the csv files `Eg: ../paths/`

| options        | Description                                                |
|----------------|------------------------------------------------------------|
| --image_folder | Folder of image patches with directory                     |
| --image_format | Image format tif (patch_gen.py save patches in tif format) |
| --label_folder | Folder of label patches with directory                     |
| --label_format | Label format tif (patch_gen.py save patches in tif format) |
| --patch_size   | Patch size to feed network. Default size is 256            |
| --output_csv   | csv filename with directory                                |

<b> Example </b>

python csv_paths.py --image_folder ../data/mass_patches/image/ --label_folder ../data/mass_patches/label/
--output_csv ../paths/data_rnb.csv

###  Training
* Training FCNs (UNet, SegNet, ResUNet and UNet-Mini) for semantic segmentation 
* For Binary classification, `--num_classes = 1`
* For Binary classification with one-hot encoding, `--num_classes = 2`
* For multi class classification, `--num_classes = number of target classes (>1)`

| options       | Description                                                                                         |
|---------------|-----------------------------------------------------------------------------------------------------|
| --model       | Name of the FCN model. Existing models are unet, unet_mini, segnet and resunet                      |
| --train_csv   | CSV file name with directory, consists of directories of image and label patches of training set.   |
| --valid_csv   | CSV file name with directory, consists of directories of image and label patches of validation set. |
| --input_shape | Input shape of model to feed patches (patch_size patch_size channels)                               |
| --batch_size  | Batch size, depends on GPU/CPU memory                                                               |
| --num_classes | Number of classes in labels data                                                                    |
| --epochs      | Number of epochs                                                                                    |
| --rs          | Radiometric resolution of the input images to rescale (Eg: 8, 12 and etc.)                          |
| --rs_label    | The value for rescaling label images (Eg: `--rs_label 255` for converting 0 & 255 values to 0 & 1   |
| --weights     | Pretrained weights file for fine tuning the model                                                   |

<b> Example </b>

python train.py --model unet_mini --train_csv ../paths/data_rnb.csv --valid_csv ../paths/data_rnb.csv
--input_shape 256 256 3 --batch_size 4 --num_classes 3 --epochs 100

###  Accuracy
* Calculates the accuracy using different accuracy metrics
* IoU, F1-Score, Precision and Recall

| options       | Description                                                                                   |
|---------------|-----------------------------------------------------------------------------------------------|
| --input_shape | Input shape of model (patch_size, patch_size, channels)                                       |
| --weights     | Trained model with directory                                                                  |
| --csv_paths   | CSV file name with directory, consists of directories of image and label patches of test set. |
| --num_classes | Number of classes in labels data                                                              |

<b> Example </b>

python accuracy.py --model unet_mini --input_shape 256 256 3 --weights ../trained_models/unet_mini_256_100_23_10_22.hdf5
--csv_paths ../paths/data_rnb.csv --num_classes 3

###  Prediction
* Predicts the entire image/tile with trained model
* Output directory has to create to save the predicted images `Eg: ../data/predictions/`

| options         | Description                                                                    |
|-----------------|--------------------------------------------------------------------------------|
| --model         | Name of the FCN model. Existing models are unet, unet_mini, segnet and resunet |
| --input_shape   | Input shape of model (patch_size, patch_size, channels)                        |
| --weights       | Trained model with directory                                                   |
| --image_folder  | Folder of input images/tiles with directory                                    |
| --image_format  | Image format tiff/tif/jpg/png                                                  |
| --output_folder | Output folder to save predicted images/tiles                                   |
 | --num_classes   | Number of classes in labels data                                               |
 | --rs            | Radiometric resolution of the input images (Eg: 8, 12 and etc.)                |

<b> Example: </b>

python tile_predict.py --model unet_mini --input_shape 256 256 3 --weights
../trained_models/unet_mini_256_100_07_11_22.hdf5 --image_folder ../data/massachusetts/test/image/
--image_format tiff --output_folder ../data/predictions/ --num_classes 3

### Summary of the Model
* Summary of FCNs
* Useful to check the configuration of FCNs, number of parameters in each layer and total
* Replace `unet, segnet and resunet` with `unet_mini` to check configuration of all networks

| options       | Description                                                                |
|---------------|----------------------------------------------------------------------------|
| --model       | Name of FCN model. Existing models are unet, unet_mini, segnet and resunet |
| --input_shape | Input shape of model to feed (patch_size patch_size channels)              |
| --num_classes | Number of classes to train                                                 |

<b> Example </b>

python summary.py --model unet_mini --input_shape 256 256 3 --num_classes 3

### For examples of other scripts, check [Google Colab Demo Notebook](https://colab.research.google.com/drive/1vPDrWKzwqIelCkZKi_GfXFdFGh4RHNL6?usp=sharing)

* `split_dataset.py` for splitting entire dataset into train, test and validation sets
* `patch_predict.py` for predicting all patches and save in the directory
* `osm_data.py` for downloading OpenStreetMap data to generate labels

### Example Outputs
<p align="center">
  <img width="900" height="1300"  src="/data/figures/mass_sota.png">
</p>

### Benchmark datasets
1. Massachusetts Benchmark datasets for Roads and Buildings extraction <br/>
[https://academictorrents.com/browse.php?search=Volodymyr
   +Mnih](https://academictorrents.com/browse.php?search=Volodymyr+Mnih)
2. List of Benchmark datasets for semantic segmentation, object detection from remote sensing imagery
[https://github.com/chrieke/awesome-satellite-imagery
   -datasets](https://github.com/chrieke/awesome-satellite-imagery-datasets)

#### Any problem with code?
Please open the [issue](https://github.com/venkanna37/Label-Pixels/issues)
