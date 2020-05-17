import argparse
import csv
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.callbacks import CSVLogger
import keras
import datetime
from models import datagen_onehot, resunet_novel2


def add_parser(subparser):
    parser = subparser.add_parser("train_onehot", help="Training model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, help="Model name, should be from unet, resunet, segnet")
    parser.add_argument("--train_csv", type=str, help="CSV file with image and label paths from training data")
    parser.add_argument("--trained_model_name", type=str, help="Trained model name, training starting time will be "
                                                               "added at the end")
    parser.add_argument("--valid_csv", type=str, help="CSV file with image and label paths from validation data")
    parser.add_argument("--input_shape", nargs='+', type=int, help="Input shape of the model (rows, columns, channels)")
    parser.add_argument("--batch_size", type=int, help="Batch size in each epoch")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.set_defaults(func=train)


def train(args):
    train_csv = args.train_csv
    valid_csv = args.valid_csv
    image_paths = []
    label_paths = []
    valid_image_paths = []
    valid_label_paths = []

    with open(train_csv, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            image_paths.append(row[0])
            label_paths.append(row[1])

    with open(valid_csv, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            valid_image_paths.append(row[0])
            valid_label_paths.append(row[1])

    model = resunet_novel2.build_res_unet(args)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    input_shape = args.input_shape
    train_gen = datagen_onehot.DataGenerator(image_paths, label_paths, batch_size=args.batch_size, n_channels=input_shape[2], patch_size=input_shape[1], shuffle=True)
    valid_gen = datagen_onehot.DataGenerator(valid_image_paths, valid_label_paths, batch_size=args.batch_size, n_channels=input_shape[2], patch_size=input_shape[1], shuffle=True)
    train_steps = len(image_paths) // args.batch_size
    valid_steps = len(valid_image_paths) // args.batch_size

    path = "./trained_models/"
    model_name = args.trained_model_name
    model_file = path + model_name + str(args.epochs) + datetime.datetime.today().strftime("_%d_%m_%y") + ".hdf5"
    log_file = path + model_name + str(args.epochs) + datetime.datetime.today().strftime("_%d_%m_%y") + ".log"
    # Training the model
    model_checkpoint = ModelCheckpoint(model_file, monitor='loss', verbose=1, save_best_only=True)
    csv_logger = CSVLogger(log_file, separator=',', append=False)
    model.fit_generator(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps,
                        epochs=args.epochs, callbacks=[model_checkpoint, csv_logger])

    # Save the model
    print("Model successfully trained")



