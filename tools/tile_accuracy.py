import gdal
import glob
from sklearn.metrics import confusion_matrix
import numpy as np

label_images = sorted(glob.glob("/home/venkanna/msc_project/test_data/deepglobe/map/*"))
pred_images = sorted(glob.glob("/home/venkanna/msc_project/test_data/deepglobe/unet_Ddeep_Wmass/*"))
print(len(label_images), len(pred_images))

tn, fp, fn, tp = 0, 0, 0, 0
x = 0
for i, j in zip(label_images, pred_images):
    label = gdal.Open(i)
    pred = gdal.Open(j)
    label_array = np.array(label.GetRasterBand(1).ReadAsArray())/255
    pred_array = np.array(pred.GetRasterBand(1).ReadAsArray())
    print(label_array.shape, pred_array.shape)
    print(label_array[0], pred_array[0])
    A = np.around(label_array.flatten())
    B = np.around(pred_array.flatten())
    cm = confusion_matrix(A, B)
    if len(cm) == 1:
        # rows.append([test_image_paths[i], test_label_paths[i], cm[0][0], 0, 0, 0])
        tn += cm[0][0]
    else:
        # rows.append([test_image_paths[i], test_label_paths[i], cm[0][0], cm[0][1], cm[1][0], cm[1][1]])
        tn += cm[0][0]
        fp += cm[0][1]
        fn += cm[1][0]
        tp += cm[1][1]
    print("Calculated " + str(x + 1) + " Images")
    x += 1
print(tn, fp, fn, tp)
iou = tp / (tp + fp + fn)
f_score = (2 * tp) / (2 * tp + fp + fn)

print("IOU Score: " + str(iou))
print("F-Score: " + str(f_score))
