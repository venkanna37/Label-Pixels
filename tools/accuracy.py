import csv
import gdal
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
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
    y = []
    y_pred = []
    for i in range(len(test_image_paths)):
        image = gdal.Open(test_image_paths[i])
        image_array = np.array(image.ReadAsArray()) / 255
        image_array = image_array.transpose(1, 2, 0)
        label = gdal.Open(test_label_paths[i])
        label_array = np.array(label.ReadAsArray())
        label_array = np.expand_dims(label_array, axis=-1)
        fm = np.expand_dims(image_array, axis=0)
        result_array = model.predict(fm)
        result_array = np.argmax(result_array[0], axis=2)
        result_array = np.squeeze(result_array)
        y.append(np.around(label_array))
        y_pred.append(result_array)
        print("Predicted " + str(i + 1) + " Images")
    # print(len(np.array(y).flatten()), len(np.array(y_pred).flatten()))
    print("\n")
    cm = confusion_matrix(np.array(y).flatten(), np.array(y_pred).flatten())
    cm_multi = multilabel_confusion_matrix(np.array(y).flatten(), np.array(y_pred).flatten())
    print("Confusion Matrix " + "\n")
    print(cm, "\n")
    accuracy = np.trace(cm/np.sum(cm))
    print("Overal Accuracy: ", round(accuracy, 3), "\n")

    mean_iou = 0
    mean_f1 = 0
    for j in range(len(cm_multi)):
        print("Class: " + str(j))
        iou = cm_multi[j][1][1] / (cm_multi[j][1][1] + cm_multi[j][0][1] + cm_multi[j][1][0])
        f1 = (2 * cm_multi[j][1][1]) / (2 * cm_multi[j][1][1] + cm_multi[j][0][1] + cm_multi[j][1][0])
        precision = cm_multi[j][1][1] / (cm_multi[j][1][1] + cm_multi[j][0][1])
        recall = cm_multi[j][1][1] / (cm_multi[j][1][1] + cm_multi[j][1][0])
        mean_iou  += iou
        mean_f1 += f1
        print("IoU Score: ", round(iou, 3))
        print("F1-Measure: ", round(f1, 3))
        print("Precision: ", round(precision, 3))
        print("Recall: ", round(recall, 3), "\n")
    print("Mean IoU Score: ", round(mean_iou/len(cm_multi), 3))
    print("Mean F1-Measure: ", round(mean_f1/len(cm_multi), 3))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Name of the model (from unet, resunet, or segnet)")
    parser.add_argument("--input_shape", nargs='+', type=int, help="Input shape of the model (rows, columns, channels)")
    parser.add_argument("--weights", type=str, help="Name and path of the trained model")
    parser.add_argument("--csv_paths", type=str, help="CSV file with image and label paths")
    parser.add_argument("--num_classes", type=int, help="Number of classes")
    args = parser.parse_args()
    accuracy(args)
