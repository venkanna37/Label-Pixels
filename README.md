##  Label-Pixels
Label-Pixels is a tool for semantic segmentation of remote sensing images using fully convolutional networks (FCNs), designed for extracting the road network from remote sensing imagery and it can be
 used in other applications applications to label every pixel in the image ( Semantic segmentation).
  This is part of my MSc research project (Automatic Road Extraction from High-Resolution Remote Sensing Imagery
  using Fully Convolutional Networks and Transfer Learning).

####  Clone repository
```commandline
git clone https://github.com/venkanna37/Label-Pixels.git
```
#### Install packages
```commandline
conda env create -f environment.yml
```
OR
```commandline
conda install -c conda-forge keras
conda install -c conda-forge gdal
conda install -c anaconda scikit-learn
```
#### Usage

<p align="center">
  <img width="900" height="180"  src="/data/methods.png">
</p>

#####  1. Patch Generation
* Generate patches from Images/Tiles
* To generate patches for train, test and valid sets, the command needs to be run three times
* Name of image and label files should be same
* Patches would be created to use data generators in KERAS and reduce the memory consumption

```commandline
usage: patch_gen.py [-h] [--image_folder IMAGE_FOLDER]
                    [--image_format IMAGE_FORMAT]
                    [--label_folder LABEL_FOLDER]
                    [--label_format LABEL_FORMAT] [--patch_size PATCH_SIZE]
                    [--overlap OVERLAP] [--output_folder OUTPUT_FOLDER]

optional arguments:
  -h, --help            show this help message and exit
  --image_folder IMAGE_FOLDER
                        Folder of input images
  --image_format IMAGE_FORMAT
                        Image format
  --label_folder LABEL_FOLDER
                        Folder of corresponding labels
  --label_format LABEL_FORMAT
                        Label format
  --patch_size PATCH_SIZE
                        Patch size to split the tiles/images
  --overlap OVERLAP     Overlap between two patches
  --output_folder OUTPUT_FOLDER
                        Output folder to save images and labels

Example:
python patch_gen.py --image_folder ..\\data\\mass_sample\\test\\image\\ --image_format tiff --label_folder ..\\data\\mass_sample\\test\\roads_and_buildings\\ --label_format tif --patch_size 256 --output_folder ..\\data\\mass_patches\\
```

##### 2. CSV Paths
* Saves location of images and labels in CSV file instead of reading patches from folders directly
```commandline
usage: csv_paths.py [-h] [--image_folder IMAGE_FOLDER]
                    [--image_format IMAGE_FORMAT]
                    [--label_folder LABEL_FOLDER]
                    [--label_format LABEL_FORMAT] [--output_csv OUTPUT_CSV]

optional arguments:
  -h, --help            show this help message and exit
  --image_folder IMAGE_FOLDER
                        Image folder
  --image_format IMAGE_FORMAT
                        Image format
  --label_folder LABEL_FOLDER
                        Label folder
  --label_format LABEL_FORMAT
                        Label format
  --output_csv OUTPUT_CSV
                        CSV file name with directory

Example:
python csv_paths.py --image_folder ..\\data\\mass_patches\\image\\ --image_format tif --label_folder ..\\data\\mass_patches\\label\\ --label_format tif --output_csv ..\\paths\\data_rd.csv
```

#####  3. Training
* Training FCNs for semantic segmentation
```commandline
usage: train.py [-h] [--model MODEL] [--train_csv TRAIN_CSV]
                [--valid_csv VALID_CSV]
                [--input_shape INPUT_SHAPE [INPUT_SHAPE ...]]
                [--batch_size BATCH_SIZE] [--num_classes NUM_CLASSES]
                [--epochs EPOCHS]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Model name (from unet, resunet, segnet)
  --train_csv TRAIN_CSV
                        CSV file with image and label paths from training data
  --valid_csv VALID_CSV
                        CSV file with image and label paths from validation
                        data
  --input_shape INPUT_SHAPE [INPUT_SHAPE ...]
                        Input shape of the model (rows, columns, channels)
  --batch_size BATCH_SIZE
                        Batch size
  --num_classes NUM_CLASSES
                        Number of classes
  --epochs EPOCHS       Number of epochs

Example:
python train.py --model unet --train_csv ..\\paths\\data_rd.csv --valid_csv ..\\paths\\data_rd.csv --input_shape 256 256 3 --batch_size 1 --num_classes 3 --epochs 100
```

#####  4. Accuracy
* Calculates the accuracy using different accuracy metrics.
```commandline
usage: accuracy.py [-h] [--model MODEL]
                   [--input_shape INPUT_SHAPE [INPUT_SHAPE ...]]
                   [--weights WEIGHTS] [--csv_paths CSV_PATHS]
                   [--num_classes NUM_CLASSES]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Name of the model (from unet, resunet, or segnet)
  --input_shape INPUT_SHAPE [INPUT_SHAPE ...]
                        Input shape of the model (rows, columns, channels)
  --weights WEIGHTS     Name and path of the trained model
  --csv_paths CSV_PATHS
                        CSV file with image and label paths
  --num_classes NUM_CLASSES
                        Number of classes

Example:
python accuracy.py --model unet --input_shape 256 256 3 --weights ..\\trained_models\\unet300_06_07_20.hdf5 --csv_paths ..\\paths\\data_rd.csv --num_classes 3
```

#####  5. Prediction
* Predicts the the entire image/tile with trained model.
```commandline
usage: tile_predict.py [-h] [--model MODEL]
                       [--input_shape INPUT_SHAPE [INPUT_SHAPE ...]]
                       [--weights WEIGHTS] [--image_folder IMAGE_FOLDER]
                       [--image_format IMAGE_FORMAT]
                       [--output_folder OUTPUT_FOLDER]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Name of the model, should be from unet, resunet,
                        segnet
  --input_shape INPUT_SHAPE [INPUT_SHAPE ...]
                        Input shape of the model (rows, columns, channels)
  --weights WEIGHTS     Name and path of the trained model
  --image_folder IMAGE_FOLDER
                        Folder of image or images
  --image_format IMAGE_FORMAT
                        Image format
  --output_folder OUTPUT_FOLDER
                        Output path of the predicted images

Example:
python tile_predict.py --model unet --input_shape 256 256 3 --weights ..\\trained_models\\unet300_06_07_20.hdf5 --image_folder ..\\data\\mass_sample\\test\\image\\ --image_format tiff --output_folder ..\\data\\
```

##### 6. Summary of the Model
* Summary of FCNs
```commandline
usage: summary.py [-h] [--model MODEL]
                  [--input_shape INPUT_SHAPE [INPUT_SHAPE ...]]
                  [--num_classes NUM_CLASSES]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Model name, should be from unet, resunet, segnet
  --input_shape INPUT_SHAPE [INPUT_SHAPE ...]
                        Input shape of the model (rows, columns, channels)
  --num_classes NUM_CLASSES
                        Number of classes in label data
```

#####  7. Rasterize
* Creating labels wiht shapefiles
```commandline
usage: rasterize.py [-h] [--raster RASTER] [--vector VECTOR] [--buffer BUFFER]
                    [--output_file OUTPUT_FILE] [--attribute ATTRIBUTE]

optional arguments:
  -h, --help            show this help message and exit
  --raster RASTER       Raster file name with directory
  --vector VECTOR       Vector file name with directory
  --buffer BUFFER       Buffer width of line feature
  --output_file OUTPUT_FILE
                        Output image name with directory
  --attribute ATTRIBUTE
                        Attribute from the vector file
Example:
python rasterize.py --raster ..\\data\\spacenet\\raster\\spacenet_chip0.tif --vector ..\\data\\spacenet\\vector\\spacenet_chip0.shp --buffer 3 --output_file ..\\data\\spacenet\\binary\\test.tif
```

#### Sample Outputs
<p align="center">
  <img width="900" height="1300"  src="/data/mass_sota.png">
  <img width="900" height="330"  src="/data/label-pixels_0001.png">
</p>

### Benchmark datasets
1. Massachusetts Benchmark datasets for Roads and Buildings extraction <br/>
[https://academictorrents.com/browse.php?search=Volodymyr+Mnih](https://academictorrents.com/browse.php?search=Volodymyr+Mnih)
2. List of Benchmark datasets for semantic segmentation, object detection from remote sensing imagery
[https://github.com/chrieke/awesome-satellite-imagery-datasets](https://github.com/chrieke/awesome-satellite-imagery-datasets)

#### Any problem with code?
Open [issue](https://github.com/venkanna37/Label-Pixels/issues) or mail me :point_right:  g.venkanna37@gmail.com
