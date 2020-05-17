import matplotlib.pyplot as plt
import gdal
import glob
import numpy as np

image_paths = sorted(glob.glob("/home/venkanna/msc_project2/images_1/mass/sat/*"))
label_paths = sorted(glob.glob("/home/venkanna/msc_project2/images_1/mass/map/*"))
unet_paths = sorted(glob.glob("/home/venkanna/msc_project2/images_1/mass/unet/*.tif"))
segnet_paths = sorted(glob.glob("/home/venkanna/msc_project2/images_1/mass/Dmass_Wdeep/*.tif"))
# resunet_paths = sorted(glob.glob("/home/venkanna/msc_project2/images_1/mass/reuse_resunet/*.tif"))
# segnet_pathsimshow(image_array) = sorted(glob.glob("/home/venkanna/msc_project/test_data/massachusetts/reuse_segnet/*.tif"))
print(len(image_paths), len(label_paths), len(unet_paths))

fig = plt.figure()
rows, columns = 4, 3
r1, r2, r3, r4, r5 = 0, 3, 6, 9, 12
for i in range(3):
    image = gdal.Open(image_paths[i])
    image_array = np.array(image.ReadAsArray())/255
    image_array = image_array.transpose(1, 2, 0)
    r1 = r1+1
    ax = fig.add_subplot(rows, columns, r1)
    ax.imshow(image_array)
    # ax.axis('off')
    ax.set_title("Tile-" + str(1+i))
    ax.axis('off')

    label = gdal.Open(label_paths[i])
    label_array = np.array(label.GetRasterBand(1).ReadAsArray())
    # label_array = np.moveaxis(label_array, 0, -1)
    r2 = r2+1
    ax1 = fig.add_subplot(rows, columns, r2)
    ax1.imshow(label_array)
    ax1.axis('off')

    un = gdal.Open(unet_paths[i])
    un_array = np.array(un.ReadAsArray())
    r3 = r3+1
    ax2 = fig.add_subplot(rows, columns, r3)
    ax2.imshow(un_array)
    ax2.axis('off')

    run = gdal.Open(segnet_paths[i])
    run_array = np.array(run.ReadAsArray())
    r4 = r4+1
    ax3 = fig.add_subplot(rows, columns, r4)
    ax3.imshow(run_array)
    ax3.axis('off')

    # sn = gdal.Open(resunet_paths[i])
    # sn_array = np.array(sn.ReadAsArray())
    # r5 = r5+1
    # ax = fig.add_subplot(rows, columns, r5)
    # ax = plt.imshow(sn_array)
    # plt.axis('off')

fig.savefig('../images_1/finetune_mass_sota.png', dpi = 1200)
# plt.xticks([])
# plt.yticks([])
plt.show()
