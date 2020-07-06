import csv
import gdal
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
import argparse
from models import resunet_model, unet_model, segnet_model


def accuracy(args):
    if args.model == "unet":
        model = unet_model.UNet(args)
        model.load_weights(args.weights)
    elif args.model == "resunet":
        # model = load_model(args.weights)
        model = resunet_model.build_res_unet(args)
        model.load_weights(args.weights)
    elif args.model == "segnet":
        model = segnet_model.create_segnet(args)
        model.load_weights(args.weights)
    else:
        print("The model name should be from the unet, resunet or segnet")
    # print(model)
    paths_file = args.csv_paths
    test_image_paths = []
    test_label_paths = []
    with open(paths_file, 'r', newline='\n') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            test_image_paths.append(row[0])
            test_label_paths.append(row[1])

    tn, fp, fn, tp = 0, 0, 0, 0
    rows = []
    for i in range(len(test_image_paths)):
        image = gdal.Open(test_image_paths[i])
        image_array = np.array(image.ReadAsArray()) / 255
        image_array = image_array.transpose(1, 2, 0)
        label = gdal.Open(test_label_paths[i])
        label_array = np.array(label.ReadAsArray())
        label_array = np.expand_dims(label_array, axis=-1)
        fm = np.expand_dims(image_array, axis=0)
        result_array = model.predict(fm)
        print(result_array[0][0][0])
        print(result_array.shape)
        result_array = np.argmax(result_array[0], axis=2)

        print(np.unique(result_array))
        result_array = np.squeeze(result_array)
        A = np.around(label_array.flatten())
        B = np.around(result_array.flatten())
        cm = confusion_matrix(A, B)
        if len(cm) == 1:
            rows.append([test_image_paths[i], test_label_paths[i], cm[0][0], 0, 0, 0])
            tn += cm[0][0]
        else:
            rows.append([test_image_paths[i], test_label_paths[i], cm[0][0], cm[0][1], cm[1][0], cm[1][1]])
            tn += cm[0][0]
            fp += cm[0][1]
            fn += cm[1][0]
            tp += cm[1][1]
        print("Predicted " + str(i + 1) + " Images")

    iou = tp / (tp + fp + fn)
    f_score = (2 * tp) / (2 * tp + fp + fn)

    print("True Possitive: " + str(tp))
    print("False Possitive: " + str(fp))
    print("True Negative: " + str(tn))
    print("False Negative: " + str(tp) + "\n")
    print("IOU Score: " + str(iou))
    print("F-Score: " + str(f_score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Name of the model (from unet, resunet, or segnet)")
    parser.add_argument("--input_shape", nargs='+', type=int, help="Input shape of the model (rows, columns, channels)")
    parser.add_argument("--weights", type=str, help="Name and path of the trained model")
    parser.add_argument("--csv_paths", type=str, help="CSV file with image and label paths")
    parser.add_argument("--num_classes", type=int, help="Number of classes")
    parser.add_argument("--onehot", type=str, help="yes or no, yes if predictions are onehot ", default="no")
    args = parser.parse_args()
    accuracy(args)
