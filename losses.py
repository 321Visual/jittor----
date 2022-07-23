import random

import jittor
import jittor.nn as nn
from vggloss import VGG19
import time
from utils import utils


class SmoothL1Loss(jittor.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def execute(self, output, target):
        return nn.smooth_l1_loss(target, output, reduction=self.reduction)


class LossesComputer():
    def __init__(self, opt):
        super(LossesComputer, self).__init__()
        self.opt = opt
        if not opt.no_labelmix:
            # self.labelmix_function = nn.MSELoss()
            self.labelmix_function = SmoothL1Loss(reduction="sum")

    def loss(self, input, label, for_real):
        # --- balancing classes ---
        weight_map = get_class_balancing(self.opt, input, label)
        # --- n+1 loss ---
        target = get_n1_target(self.opt, input, label, for_real)
        loss = nn.cross_entropy_loss(input, target, reduction='none')
        if for_real:
            loss = jittor.mean(loss * weight_map[:, 0, :, :])
        else:
            loss = jittor.mean(loss)
        return loss

    def loss_labelmix(self, mask, output_D_mixed, output_D_fake, output_D_real):
        mixed_D_output = mask * output_D_real + (1 - mask) * output_D_fake
        return self.labelmix_function(mixed_D_output, output_D_mixed)


def get_class_balancing(opt, input, label):
    if not opt.no_balancing_inloss:
        class_occurence = jittor.sum(label, dims=tuple([0, 2, 3]))
        if opt.contain_dontcare_label:
            class_occurence[0] = 0
        num_of_classes = (class_occurence > 0).sum()
        coefficients = (1 / class_occurence) * label.numel() / (num_of_classes * label.shape[1])
        integers = jittor.argmax(label, dim=1, keepdims=True)[0]
        if opt.contain_dontcare_label:
            coefficients[0] = 0
        weight_map = coefficients[integers]
    else:
        weight_map = jittor.ones_like(input[:, :, :, :])
    return weight_map


def get_n1_target(opt, input, label, target_is_real):
    targets = get_target_tensor(opt, input, target_is_real)
    num_of_classes = label.shape[1]
    integers = jittor.argmax(label, dim=1)[0]
    targets = targets[:, 0, :, :] * num_of_classes
    integers += targets.long()
    integers = jittor.clamp(integers, min_v=num_of_classes) - num_of_classes
    return integers


def get_target_tensor(opt, input, target_is_real):
    if target_is_real:
        # return jittor.full(input.shape, val=random.uniform(0.95, 1.05), dtype="float32")
        return jittor.full(input.shape, val=1.0, dtype="float32")
    else:
        return jittor.full(input.shape, val=random.uniform(0.0, 0.05), dtype="float32")
        # return jittor.full(input.shape, val=0.0, dtype="float32")


def generate_labelmix(label, fake_image, real_image, opt):
    target_map = jittor.argmax(label, dim=1, keepdims=True)[0]
    all_classes = jittor.unique(target_map)
    for c in all_classes:
        target_map[target_map == c] = jittor.randint(0, 2, (1,))
    target_map = target_map.float()
    mixed_image = target_map * real_image + (1 - target_map) * fake_image
    return mixed_image, target_map


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def execute(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
