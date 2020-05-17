import argparse
import csv
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
import keras
import datetime
from models import unet_model, resunet_novel, segnet_model, datagen_with_pred
import sys
import numpy as np


def train(args):
    train_csv = args.train_csv
    valid_csv = args.valid_csv
    image_paths = []
    label_paths = []
    pred_paths = []
    valid_image_paths = []
    valid_label_paths = []
    valid_pred_paths = []

    with open(train_csv, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            image_paths.append(row[0])
            label_paths.append(row[1])
            pred_paths.append(row[2])

    with open(valid_csv, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            valid_image_paths.append(row[0])
            valid_label_paths.append(row[1])
            valid_pred_paths.append(row[2])

    if args.model == "unet":
        model = unet_model.UNet(args)
        model.layers[1].name = 'test1'
        model.load_weights(args.weights, by_name=True)

    elif args.model == "resunet":
        model = resunet_model.build_res_unet(args)
        model.layers[1].name = 'test1'
        model.layers[3].name = 'test2'
        model.load_weights(args.weights, by_name=True)

    elif args.model == "segnet":
        model = segnet_model.create_segnet(args)
        model.layers[1].name = 'test1'
        model.load_weights(args.weights, by_name=True)
    else:
        print("The model name should be from the unet, resunet or segnet")

    # print(model.layers[6].get_weights())
    # for i in model.layers:
    #     print(i.name)
    # trained_model = load_model(args.weights)
    # print(trained_model.layers[6].get_weights())
    # weights = trained_model.get_weights()
    # layers = trained_model.layers
    # print(len(layers))
    # w1 = np.array(model.layers[6].get_weights)
    # w2 = np.array(trained_model.layers[6].get_weights)
    # print(w1)
    # print(w2)
    # print(model.get_weights())
    # for i, j in zip(range(5, len(model.layers)-1), range(5, len(trained_model.layers)-1)):
    #     weights1 = model.layers[i].get_weights()
    #     weights2 = trained_model.layers[j].get_weights()
    #     # print(layer1.name, np.array(weights1).shape, np.array(weights2).shape)
    #     if np.array(weights1).shape == np.array(weights2).shape:
    #         model.layers[i].set_weights(weights2)
    # # w3 = model.layers[6].get_weights
    # print(model.layers[6].get_weights())
    # sys.exit()
    # model = load_model(args.weights)
    # adm = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    train_gen = datagen_with_pred.DataGenerator(image_paths, label_paths, pred_paths, batch_size=args.batch_size, n_channels=args.channels, patch_size=args.patch_size, shuffle=True)
    valid_gen = datagen_with_pred.DataGenerator(valid_image_paths, valid_label_paths, valid_pred_paths, batch_size=args.batch_size, n_channels=args.channels, patch_size=args.patch_size, shuffle=True)
    train_steps = len(image_paths) // args.batch_size
    valid_steps = len(valid_image_paths) // args.batch_size

    path = "../trained_models/"
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--weights", type=str, help="Trained model")
    parser.add_argument("--train_csv", type=str, help="CSV file with image and label paths from training data")
    parser.add_argument("--valid_csv", type=str, help="CSV file with image and label paths from validation data")
    parser.add_argument("--trained_model_name", type=str, help="Trained model name, training starting time will be "
                                                               "added at the end")
    parser.add_argument("--input_shape", nargs='+', type=int, help="Input shape of the model (rows, columns, channels)")
    parser.add_argument("--batch_size", type=int, help="Batch size in each epoch")
    parser.add_argument("--channels", type=int, help="Batch size in each epoch")
    parser.add_argument("--patch_size", type=int, help="Batch size in each epoch")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    args = parser.parse_args()
    train(args)
