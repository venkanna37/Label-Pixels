import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import accuracy_score
from models import lp_utils as lu
import argparse


def cnn_predict(args):
    test_image_paths, test_label = lu.file_paths(args.csv_paths)
    # test_image_paths, test_label = test_image_paths[:200], test_label[:200]
    model = load_model(args.weights)
    predictions = []
    for i, img in enumerate(test_image_paths):
        _img = image.load_img(img, target_size=(224, 224))
        _img = image.img_to_array(_img)
        _img = np.expand_dims(_img, axis=0)
        _img = preprocess_input(_img)
        pred = model.predict(_img)
        final_label = np.argmax(pred)
        predictions.append(final_label)
        print(i)
    y = np.array(test_label).astype(np.int)
    y_pred = np.array(predictions)
    acc = accuracy_score(y, y_pred)
    print("Overal Accuracy: ", round(acc, 3), "\n")
    cm = confusion_matrix(y, y_pred)
    print("Confusion Matrix:")
    print(cm, "\n")
    cm_multi = multilabel_confusion_matrix(y, y_pred)
    for j in range(len(cm_multi)):
        print("Class: " + str(j))
        iou = cm_multi[j][1][1] / (cm_multi[j][1][1] + cm_multi[j][0][1] + cm_multi[j][1][0])
        f1 = (2 * cm_multi[j][1][1]) / (2 * cm_multi[j][1][1] + cm_multi[j][0][1] + cm_multi[j][1][0])
        precision = cm_multi[j][1][1] / (cm_multi[j][1][1] + cm_multi[j][0][1])
        recall = cm_multi[j][1][1] / (cm_multi[j][1][1] + cm_multi[j][1][0])
        print("IoU Score: ", round(iou, 3))
        print("F1-Measure: ", round(f1, 3))
        print("Precision: ", round(precision, 3))
        print("Recall: ", round(recall, 3), "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Name of the model (from unet, resunet, or segnet)")
    parser.add_argument("--input_shape", nargs='+', type=int, help="Input shape of the model (rows, columns, channels)")
    parser.add_argument("--weights", type=str, help="Name and path of the trained model")
    parser.add_argument("--csv_paths", type=str, help="CSV file with image and label paths")
    parser.add_argument("--num_classes", type=int, help="Number of classes")
    args = parser.parse_args()
    cnn_predict(args)
