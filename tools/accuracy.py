"""
Accuracy of the model
 - Calculates from the patches generated with patch_gen script
 - Patch locations should be saved in csv file (cvs_paths script)
"""

import gdal
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
import argparse
from models import lp_utils as lu


def accuracy(model_name, weights_path, input_shape, num_classes, csv_paths, rs, rs_label):
    # Loading model with weights
    model = lu.select_model(model_name, weights_path, input_shape, num_classes)
    # Saving image paths as a list
    test_image_paths, test_label_paths = lu.file_paths(csv_paths)
    # Calculating accuracy
    y, y_pred = [], []
    rescale_value = lu.rescaling_value(rs)
    for i in range(len(test_image_paths)):
        image = gdal.Open(test_image_paths[i])
        image_array = np.array(image.ReadAsArray()) / rescale_value
        image_array = image_array.transpose(1, 2, 0)
        label = gdal.Open(test_label_paths[i])
        label_array = np.array(label.ReadAsArray()) / rs_label
        label_array = np.expand_dims(label_array, axis=-1)
        fm = np.expand_dims(image_array, axis=0)
        result_array = model.predict(fm)
        if num_classes != 1:
            result_array = np.argmax(result_array[0], axis=2)
        result_array = np.squeeze(result_array)
        if num_classes == 1:
            result_array = np.around(result_array.flatten())
        y.append(np.around(label_array))
        y_pred.append(result_array)
        print("Predicted " + str(i + 1) + " Patches")
    print("\n")
    print("list of classes from predictions: " + str(np.unique(np.array(y_pred))))
    print("list of classes from labels: " + str(np.unique(np.array(y))))
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
        mean_iou += iou
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
    parser.add_argument("--rs", type=int, help="Radiometric resolution of the image", default=8)
    parser.add_argument("--rs_label", type=int, help="Rescaling labels if they are not single digits", default=1)
    args = parser.parse_args()

    accuracy(args.model, args.weights, tuple(args.input_shape), args.num_classes,
             args.csv_paths, args.rs, args.rs_label)
