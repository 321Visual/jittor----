import glob
import random
import jittor
from jittor import transform as TR
from jittor.dataset.dataset import Dataset
import os
from PIL import Image
import numpy as np
import cv2


class ImageDataset(Dataset):
    def __init__(self, opt, mode="train"):
        super(ImageDataset, self).__init__()
        opt.crop_size = opt.img_width
        opt.label_nc = 29
        opt.contain_dontcare_label = False
        opt.semantic_nc = 29  # label_nc + unknown
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 512 / 384
        self.mode = mode
        self.opt = opt
        self.images, self.labels = self.list_images()
        self.normalize = TR.ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.set_attrs(total_len=len(self.labels))
        print(len(self.labels))

    def __getitem__(self, idx):
        idx = idx % len(self.labels)
        if not self.mode.startswith("test"):
            image = Image.open(self.images[idx]).convert('RGB')
        else:
            image = "None"
        label = Image.open(self.labels[idx])
        image, label = self.transforms(label=label, image=image)
        return {"image": image, "label": label, "name": self.labels[idx]}

    def list_images(self):
        def get_filelist(path):
            Filelist = []
            for home, dirs, files in os.walk(path):
                for filename in files:
                    Filelist.append(os.path.join(home, filename))
            return Filelist

        if self.mode.startswith("test"):
            path_img = None
            images = None
            path_lab = self.opt.input_path
            labels = sorted(get_filelist(path_lab))
        elif self.mode.startswith("val"):
            # path_img = os.path.join(self.opt.dataroot, self.mode, "imgs")
            path_img = os.path.join(self.opt.input_path, "imgs")
            images = sorted(get_filelist(path_img))
            images = images[0:10]
            # path_lab = os.path.join(self.opt.dataroot, self.mode, "labels")
            path_lab = os.path.join(self.opt.input_path, "labels")
            labels = sorted(get_filelist(path_lab))
            labels = labels[0:10]
        else:
            # path_img = os.path.join(self.opt.dataroot, self.mode, "imgs")
            path_img = os.path.join(self.opt.input_path, "imgs")
            images = sorted(get_filelist(path_img))
            # path_lab = os.path.join(self.opt.dataroot, self.mode, "labels")
            path_lab = os.path.join(self.opt.input_path, "labels")
            labels = sorted(get_filelist(path_lab))
        if not self.mode.startswith("test"):
            assert len(images) == len(labels), "different len of images and labels %s - %s" % (len(images), len(labels))
            for i in range(len(images)):
                assert os.path.splitext(os.path.basename(images[i]))[0] == \
                       os.path.splitext(os.path.basename(labels[i]))[
                           0], '%s and %s are not matching %s' % (
                    images[i], labels[i], os.path.splitext(images[i]))
        return images, labels

    def transforms(self, label, image):
        new_height, new_width = (self.opt.img_height, self.opt.img_width)
        if self.mode == "train" or self.mode == "augmentation":
            assert image.size == label.size
            image = TR.function_pil.resize(image, (new_height, new_width), Image.BICUBIC)
        label = TR.function_pil.resize(label, (new_height, new_width), Image.NEAREST)
        # flip
        if not (self.mode == "val" or self.mode.startswith("test") or self.opt.no_flip):
            if random.random() < 0.5:
                image = TR.function_pil.hflip(image)
                label = TR.function_pil.hflip(label)
        # to tensor
        if not self.mode.startswith("test"):
            image = TR.to_tensor(image)
            # normalize
            image = self.normalize(image)
        label = TR.to_tensor(label)

        return image, label
