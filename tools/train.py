"""
Training Fully Convolutional Networks
"""
import argparse
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
import datetime
from models import lp_utils as lu
from models import datagen


def train(args):
    # Saving image paths as a list
    image_paths, label_paths = lu.file_paths(args.train_csv)
    valid_image_paths, valid_label_paths = lu.file_paths(args.valid_csv)
    # radiometric resolution
    rs = lu.rescaling_value(args.rs)
    # Loading model with weights
    model = lu.select_model(args)
    if args.num_classes > 1:
        loss_fun = "categorical_crossentropy"
    elif args.num_classes == 1:
        loss_fun = "binary_crossentropy"
    else:
        print("Number of classes not specified")
    model.compile(optimizer="adam", loss=loss_fun, metrics=["acc"])
    input_shape = args.input_shape
    train_gen = datagen.DataGenerator(image_paths, label_paths, batch_size=args.batch_size, n_classes=args.num_classes,
                                      n_channels=input_shape[2], patch_size=input_shape[1], shuffle=True, rs=rs,
                                      rs_label=args.rs_label)
    valid_gen = datagen.DataGenerator(valid_image_paths, valid_label_paths, batch_size=args.batch_size,
                                      n_classes=args.num_classes, n_channels=input_shape[2], patch_size=input_shape[1],
                                      shuffle=True, rs=rs, rs_label=args.rs_label)
    train_steps = len(image_paths) // args.batch_size
    valid_steps = len(valid_image_paths) // args.batch_size
    model_name = args.model
    model_file = "../trained_models/" + model_name + str(args.epochs) + datetime.datetime.today().strftime("_%d_%m_%y")\
                 + ".hdf5"
    log_file = "../trained_models/" + model_name + str(args.epochs) + datetime.datetime.today().strftime("_%d_%m_%y")\
               + ".csv"
    # Training the model
    model_checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True)
    csv_logger = CSVLogger(log_file, separator=',', append=False)
    model.fit_generator(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps,
                        epochs=args.epochs, callbacks=[model_checkpoint, csv_logger])
    print("Model successfully trained")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model name (from unet, resunet, segnet)")
    parser.add_argument("--train_csv", type=str, help="CSV file with image and label paths from training data")
    parser.add_argument("--valid_csv", type=str, help="CSV file with image and label paths from validation data")
    parser.add_argument("--input_shape", nargs='+', type=int, help="Input shape of the model (rows, columns, channels)")
    parser.add_argument("--batch_size", type=int, help="Batch size", default=4)
    parser.add_argument("--num_classes", type=int, help="Number of classes")
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=100)
    parser.add_argument("--rs", type=int, help="Radiometric resolution of the image", default=8)
    parser.add_argument("--rs_label", type=int, help="Rescaling labels if they are not single digits", default=1)
    parser.add_argument("--weights", type=str, help="Name and path of the trained model")
    args = parser.parse_args()
    train(args)
