import glob
import random
import os

valid_images = glob.glob("/media/venkanna/f03f886e-ea07-4f60-9f48-ad861022a20e/venkanna/data/deep_sample/valid/sat/*")
sample_labels = glob.glob("/media/venkanna/f03f886e-ea07-4f60-9f48-ad861022a20e/venkanna/data/deep_sample/train/map/*")
# test = random.sample(sample_images, 36)
# print(os.path.basename(sample[0]))
# for x in test:
#     file_name = os.path.basename(x)
#     os.system("mv " + x + " /media/venkanna/f03f886e-ea07-4f60-9f48-ad861022a20e/venkanna/data/deep_sample/test/sat/" + file_name)
i = 0
for x in valid_images:
    file_name = os.path.basename(x)
    file_name = file_name[:-8]
    for y in sample_labels:
        file_name2 = os.path.basename(y)
        file_name3 = file_name2[:-9]
        if file_name==file_name3:
            i = i + 1
            print(i)
            os.system("mv " + y + " /media/venkanna/f03f886e-ea07-4f60-9f48-ad861022a20e/venkanna/data/deep_sample/valid/map/" + file_name2)
