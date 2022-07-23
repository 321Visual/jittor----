import os
from glob import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import shutil
import matplotlib


def get_filelist(path):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            Filelist.append(os.path.join(home, filename))
    return Filelist


data_root = '/data/temp/data/train'
data_augmentation_dir_name = "augmentation"
path_img = os.path.join(data_root, "imgs")
path_lab = os.path.join(data_root, "labels")

# os.makedirs(os.path.join(path_img, data_augmentation_dir_name))
# os.makedirs(os.path.join(path_lab, data_augmentation_dir_name))

img_list = sorted(get_filelist(path_img))
label_list = sorted(get_filelist(path_lab))
assert len(img_list) == len(label_list)

classes_name = ["mountain", "sky", "water", "sea", "rock", "tree", "earth", "hill", "river", "sand", "land", "building",
                "grass", "plant", "person", "boat", "waterfall", "wall", "pier", "path", "lake", "bridge", "field",
                "road", "railing", "fence", "ship", "house", "other"]

# classes_cllections = np.zeros(29, dtype=np.int32)
classes_cllections = [9505, 9675, 5293, 4066, 3858, 3824, 2788, 483, 558, 1503, 1510, 1771, 1028, 939,
                      1044, 1131, 288, 332, 578, 314, 245, 170, 105, 176, 103, 230, 84, 172,
                      774]
for i, (img_file, lab_file) in enumerate(zip(img_list, label_list)):
    # img_file = os.path.join(path_img, img)
    # lab_file = os.path.join(path_lab, lab)
    image = cv2.imread(img_file)
    label = cv2.imread(lab_file, flags=cv2.IMREAD_ANYDEPTH)
    label_t = np.reshape(label, (-1))
    classes = np.unique(label_t)
    # if classes[0] is None:
    #     break
    print(i, classes, sep=":")
    # t = 0
    # if len(classes) > 5:
    #     t += 1

    for cls in classes:
        classes_cllections[cls] += 1

    #     if classes_cllections[cls] < 500:
    #         t += 2
    #         continue
    #     if classes_cllections[cls] < 200:
    #         t += 4
    #         continue
    #     if classes_cllections[cls] < 100:
    #         t += 6
    #         continue
    # for j in range(0, t):
    #     name = f"augmentation_{img.split('.')[0]}_{i}_{j}"
    #     print(name)
    #     shutil.copyfile(img_file, os.path.join(path_img, data_augmentation_dir_name, name + ".jpg"))
    #     shutil.copyfile(lab_file, os.path.join(path_lab, data_augmentation_dir_name, name + ".png"))

print("\n\n")
print(classes_cllections)

# result:
