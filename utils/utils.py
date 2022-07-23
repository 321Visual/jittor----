from typing import List

import jittor
import numpy as np
import random
import time
import os
import matplotlib.pyplot as plt
from PIL import Image
from jittor import init
import cv2
from tqdm import tqdm


@jittor.single_process_scope()
def print_parameter_count(net_list: List):
    for network in net_list:
        param_count = 0
        for name, module in network.named_modules():
            if (isinstance(module, jittor.nn.Conv2d)
                    or isinstance(module, jittor.nn.Linear)
                    or isinstance(module, jittor.nn.Embedding)):
                param_count += sum([p.numel() for p in module.parameters()])
        print('Created', network.__class__.__name__, "with %d parameters" % param_count)


def fix_seed(seed):
    random.seed(seed)
    jittor.set_global_seed(seed)
    jittor.misc.set_global_seed(seed=seed, different_seed_for_mpi=False)


def get_start_iters(start_iter, dataset_size):
    if start_iter == 0:
        return 0, 0
    start_epoch = (start_iter + 1) // dataset_size
    start_iter = (start_iter + 1) % dataset_size
    return start_epoch, start_iter


# class results_saver():
#     def __init__(self, opt):
#         path = os.path.join(opt.results_dir, opt.name, opt.ckpt_iter)
#         self.path_label = os.path.join(path, "label")
#         self.path_image = os.path.join(path, "image")
#         self.path_to_save = {"label": self.path_label, "image": self.path_image}
#         os.makedirs(self.path_label, exist_ok=True)
#         os.makedirs(self.path_image, exist_ok=True)
#         self.num_cl = opt.label_nc + 2
#
#     def __call__(self, label, generated, name):
#         assert len(label) == len(generated)
#         for i in range(len(label)):
#             im = tens_to_lab(label[i], self.num_cl)
#             self.save_im(im, "label", name[i])
#             im = tens_to_im(generated[i]) * 255
#             self.save_im(im, "image", name[i])
#
#     def save_im(self, im, mode, name):
#         im = Image.fromarray(im.astype(np.uint8))
#         im.save(os.path.join(self.path_to_save[mode], name.split("/")[-1]).replace('.jpg', '.png'))

#
# class timer():
#     def __init__(self, opt):
#         self.prev_time = time.time()
#         self.prev_epoch = 0
#         self.num_epochs = opt.num_epochs
#         self.file_name = os.path.join(opt.checkpoints_dir, opt.name, "progress.txt")
#
#     def __call__(self, epoch, cur_iter):
#         if cur_iter != 0:
#             avg = (time.time() - self.prev_time) / (cur_iter - self.prev_epoch)
#         else:
#             avg = 0
#         self.prev_time = time.time()
#         self.prev_epoch = cur_iter
#
#         with open(self.file_name, "a") as log_file:
#             log_file.write('[epoch %d/%d - iter %d], time:%.3f \n' % (epoch, self.num_epochs, cur_iter, avg))
#         print('[epoch %d/%d - iter %d], time:%.3f' % (epoch, self.num_epochs, cur_iter, avg))
#         return avg


class losses_saver():
    def __init__(self, opt):
        self.name_list = ["Generator", "Vgg", "D_fake", "D_real", "LabelMix"]
        self.opt = opt
        self.freq_smooth_loss = opt.freq_smooth_loss
        self.freq_save_loss = opt.freq_save_loss
        self.losses = dict()
        self.cur_estimates = np.zeros(len(self.name_list))
        self.path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses")
        self.is_first = True
        os.makedirs(self.path, exist_ok=True)
        for name in self.name_list:
            if opt.continue_train:
                self.losses[name] = np.load(self.path + "/losses.npy", allow_pickle=True).item()[name]
            else:
                self.losses[name] = list()

    def __call__(self, epoch, losses):
        for i, loss in enumerate(losses):
            if loss is None:
                self.cur_estimates[i] = None
            else:
                self.cur_estimates[i] += loss.numpy()
        if epoch % self.freq_smooth_loss == self.freq_smooth_loss - 1:
            for i, loss in enumerate(losses):
                if not self.cur_estimates[i] is None:
                    self.losses[self.name_list[i]].append(self.cur_estimates[i] / self.opt.freq_smooth_loss)
                    self.cur_estimates[i] = 0
        if epoch % self.freq_save_loss == self.freq_save_loss - 1:
            self.plot_losses()
            np.save(os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses", "losses"), self.losses)

    def plot_losses(self):
        for curve in self.losses:
            fig, ax = plt.subplots(1)
            n = np.array(range(len(self.losses[curve]))) * self.opt.freq_smooth_loss
            plt.plot(n[1:], self.losses[curve][1:])
            plt.ylabel('loss')
            plt.xlabel('epochs')

            plt.savefig(os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses", '%s.png' % (curve)), dpi=600)
            plt.close(fig)

        fig, ax = plt.subplots(1)
        for curve in self.losses:
            if np.isnan(self.losses[curve][0]):
                continue
            plt.plot(n[1:], self.losses[curve][1:], label=curve)
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses", 'combined.png'), dpi=600)
        plt.close(fig)


def update_EMA(netEMA, netG, cur_iter, dataloader, opt, force_run_stats=False):
    # update weights based on new generator weights
    with jittor.no_grad():
        state_EMA = netEMA.state_dict()
        state_G = netG.state_dict()
        for key in state_EMA:
            state_EMA[key] = jittor.copy(
                state_EMA[key] * opt.EMA_decay +
                state_G[key] * (1 - opt.EMA_decay)
            )
    if jittor.rank == 0:
        print("updated EMA")
    # collect running stats for batchnorm before FID computation, image or network saving
    condition_run_stats = (force_run_stats or
                           # cur_iter % opt.freq_print == 0 or
                           cur_iter % opt.freq_fid == 0
                           # cur_iter % opt.freq_save_ckpt == 0 or
                           # cur_iter % opt.freq_save_latest == 0
                           )
    if condition_run_stats:
        print(f"EMA condition run stats")
        with jittor.no_grad():
            num_upd = 0
            for i, data_i in enumerate(dataloader):
                image, label = preprocess_input(opt, data_i)
                fake = netEMA(label)
                num_upd += 1
                if num_upd > 3:
                    break


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init.kaiming_normal_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias, 0.0)
    elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        init.kaiming_normal_(var=m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias, 0.0)


def load_checkpoints_for_train(opt, generator, discriminator, netEMA=None):
    which_iter = "latest2" if opt.which_iter == -1 else opt.which_iter
    print(f"load models {which_iter}")
    path = os.path.join(opt.checkpoints_dir, opt.name, "models", str(which_iter) + "_")
    generator.load(path + "G.pkl")
    discriminator.load(path + "D.pkl")
    if not opt.no_EMA:
        print("load ema---")
        # netEMA.load(os.path.join(opt.checkpoints_dir, opt.name, "models", "42515_G.pkl"))
        # netEMA.load(path + "G.pkl")
        netEMA.load(path + "EMA.pkl")


def save_networks(opt, cur_iter, netG, netD, netEMA=None, latest=False, best=False):
    path = os.path.join(opt.checkpoints_dir, opt.name, "models")
    os.makedirs(path, exist_ok=True)
    if latest:
        print("\n-------save latest model-------\n")
        netG.save(path + '/%s_G.pkl' % "latest")
        netD.save(path + '/%s_D.pkl' % "latest")
        if not opt.no_EMA:
            netEMA.save(path + '/%s_EMA.pkl' % "latest")
        with open(os.path.join(opt.checkpoints_dir, opt.name) + "/latest_iter.txt", "w") as f:
            f.write(str(cur_iter))
    elif best:
        print("\n-------save best model-------\n")
        netG.save(path + '/%s_G.pkl' % "best")
        netD.save(path + '/%s_D.pkl' % "best")
        if not opt.no_EMA:
            netEMA.save(path + '/%s_EMA.pkl' % "best")
        with open(os.path.join(opt.checkpoints_dir, opt.name) + "/best_iter.txt", "w") as f:
            f.write(str(cur_iter))
    else:
        print("\n-------save {} model-------\n".format(cur_iter))
        netG.save(path + '/%d_G.pkl' % cur_iter)
        netD.save(path + '/%d_D.pkl' % cur_iter)
        if not opt.no_EMA:
            netEMA.save(path + '/%d_EMA.pkl' % cur_iter)


def preprocess_input(opt, data):
    label_map = data['label']
    bs, _, h, w = label_map.size()
    nc = opt.semantic_nc
    input_label = jittor.zeros(shape=(bs, nc, h, w), dtype="float32")
    input_label.scatter_(dim=1, index=label_map, src=jittor.array(1.0))
    return data['image'], input_label


@jittor.single_process_scope()
class image_saver:
    def __init__(self, opt, generator, netEMA, val_dataloader, ):
        self.path = os.path.join(opt.checkpoints_dir, opt.name, "images") + "/"
        self.opt = opt
        self.generator = generator
        self.netEMA = netEMA
        self.val_dataloader = val_dataloader
        os.makedirs(self.path, exist_ok=True)

    @jittor.single_process_scope()
    def visualize_batch(self, cur_iter):
        print(f"\n>>visualize_batch {cur_iter}<<\n")
        # self.generator.eval()
        for i, data_i in enumerate(self.val_dataloader):
            _, label_t = preprocess_input(self.opt, data_i)
            fake_B = self.generator(label_t)
            fake_B = ((fake_B + 1) / 2 * 255).numpy().astype('uint8')
            if i == 0:
                self.save_image(fake_B, f"{self.path}/iter_{cur_iter}_sample.png", nrow=3)
            break
        # self.generator.train()
        if not self.opt.no_EMA:
            # self.netEMA.eval()
            for i, data_i in enumerate(self.val_dataloader):
                _, label_t = preprocess_input(self.opt, data_i)
                fake_B = self.netEMA(label_t)
                fake_B = ((fake_B + 1) / 2 * 255).numpy().astype('uint8')
                if i == 0:
                    self.save_image(fake_B, f"{self.path}/iter_{cur_iter}_netEMA_sample.png", nrow=3)
                break
            # self.netEMA.train()

    @staticmethod
    def save_image(img, path, nrow=10):
        N, C, W, H = img.shape
        if N % nrow != 0:
            print("save_image error: N%nrow!=0")
            return
        img = img.transpose((1, 0, 2, 3))
        ncol = int(N / nrow)
        img2 = img.reshape([img.shape[0], -1, H])
        img = img2[:, :W * ncol, :]
        for i in range(1, int(img2.shape[1] / W / ncol)):
            img = np.concatenate([img, img2[:, W * ncol * i:W * ncol * (i + 1), :]], axis=2)
        img = img.transpose((1, 2, 0))
        if C == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(path, img)
        return img


def tens_to_im(tens):
    out = (tens + 1) / 2
    out = jittor.clamp(out, 0, 1)
    return np.transpose(out.numpy(), (1, 2, 0))


def tens_to_lab(tens, num_cl):
    label_tensor = Colorize(tens, num_cl)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy


###############################################################################
# Code below from
# https://github.com/visinf/1-stage-wseg/blob/38130fee2102d3a140f74a45eec46063fcbeaaf8/datasets/utils.py
# Modified so it complies with the Cityscapes label map colors (fct labelcolormap)
###############################################################################

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def Colorize(raw_label):
    cmap = np.array(
        [(111, 74, 0), (81, 0, 81), (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70),
         (102, 102, 156), (190, 153, 153),
         (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153),
         (250, 170, 30), (220, 220, 0),
         (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142),
         (0, 0, 70),
         (0, 60, 100), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)],
        dtype=np.uint8)
    # cmap = jittor.array(cmap[:num_cl], dtype=cmap.dtype)  # todo

    size = raw_label.shape
    color_image = np.zeros(shape=(size[0], 3, size[2], size[3]), dtype="uint8")
    # tens = jittor.argmax(tens, dim=0, keepdims=True)[0]
    for label in range(0, len(cmap)):
        mask = (label == raw_label)
        mask = np.squeeze(mask)
        color_image[:, 0, :, :][mask] = cmap[label][0]
        color_image[:, 1, :, :][mask] = cmap[label][1]
        color_image[:, 2, :, :][mask] = cmap[label][2]

    return color_image


#
def labelcolormap(N):
    if N == 31:
        cmap = np.array(
            [(111, 74, 0), (81, 0, 81), (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70),
             (102, 102, 156), (190, 153, 153),
             (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153),
             (250, 170, 30), (220, 220, 0),
             (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142),
             (0, 0, 70),
             (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)],
            dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i + 1  # let's give 0 a color
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap
