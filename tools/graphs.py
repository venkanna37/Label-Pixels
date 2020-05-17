import matplotlib.pyplot as plt
import csv
import argparse


def loss_acc_graph(file1, file2, file3):
    # let's visualize the learning curve of the pre-trained network
    loss = []
    epoch = []
    val_loss = []
    loss2 = []
    epoch2 = []
    val_loss2 = []
    loss3 = []
    epoch3 = []
    val_loss3 = []
    with open(file1, "r") as f:
        next(f)
        plots = csv.reader(f, delimiter=',')
        for row in plots:
            loss.append(float(row[2]))
            epoch.append(float(row[0]))
            val_loss.append(float(row[4]))

    with open(file2, "r") as g:
        next(g)
        plots = csv.reader(g, delimiter=',')
        for row in plots:
            loss2.append(float(row[2]))
            epoch2.append(float(row[0]))
            val_loss2.append(float(row[4]))

    with open(file3, "r") as h:
        next(h)
        plots = csv.reader(h, delimiter=',')
        for row in plots:
            loss3.append(float(row[2]))
            epoch3.append(float(row[0]))
            val_loss3.append(float(row[4]))

    plt.plot(epoch, loss, color='red', label="U-Net, Training loss")
    plt.plot(epoch, val_loss, color='red', label="U-Net, Validation loss")
    plt.plot(epoch2, loss2, color='y', label="SegNet, Training loss")
    plt.plot(epoch2, val_loss2, color='y', label="SegNet, Validation loss")
    plt.plot(epoch3, loss3, color='green', label= "ResUNet, Training loss")
    plt.plot(epoch3, val_loss3, color='green', label= "ResUNet, Validation loss")
    plt.title("Epoch vs Loss graph")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()


# loss_acc_graph("../trained_models/unet_mass_finetuning_with_pred_50_11_03_20.log","../trained_models/segnet_mass_finetuning_with_pred_50_16_03_20.log", "../trained_models/mass_finetuning_with_pred50_27_01_20.log" )
loss_acc_graph("../trained_models/unet_mass_256_300_05_03_20.log","../trained_models/segnet_mass_256_300_12_03_20.log", "../trained_models/resunet_mass_256_300_27_12_19.log" )
# loss_acc_graph("../trained_models/unet_deep_256_300_06_03_20.log","../trained_models/segnet_deep_256_300_07_03_20.log", "../trained_models/resunet_deep_256_300_20_12_19.log" )
# loss_acc_graph("../trained_models/unet_Ddeep_Wmass_50_11_03_20.log")
# loss_acc_graph("../trained_models/unet_Ddeep_Wmass_50_11_03_20.log",
#                "../trained_models/segnet_mass_finetuning_with_pred_50_16_03_20.log")
