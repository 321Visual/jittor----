import os
import jittor as jt

from jittor import nn
import datetime
import time
import datasets
from discriminator import OASIS_Discriminator
from generator import OASIS_Generator
from utils import utils
import losses
import copy
import warnings
from config import read_arguments

warnings.filterwarnings("ignore")
# 配置jittor相关参数
jt.cudnn.set_max_workspace_ratio(0.0)
jt.flags.use_cuda = 1

opt = read_arguments(train=True)



os.makedirs(f"{opt.checkpoints_dir}/images/", exist_ok=True)
os.makedirs(f"{opt.checkpoints_dir}/saved_models/", exist_ok=True)

dataloader = datasets.ImageDataset(opt, mode="train").set_attrs(
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
    drop_last=True,
)

val_dataloader = datasets.ImageDataset(opt, mode="val").set_attrs(
    batch_size=6,
    shuffle=True,
    num_workers=opt.n_cpu,
    drop_last=True
)

# Loss functions
losses_computer = losses.LossesComputer(opt=opt)
VGG_loss = losses.VGGLoss()

# Initialize generator and discriminator
generator = OASIS_Generator(opt=opt)
discriminator = OASIS_Discriminator(opt=opt)
with jt.no_grad():
    netEMA = copy.deepcopy(generator) if not opt.no_EMA else None
# 计算网络参数量并打印
utils.print_parameter_count([generator, discriminator])

# 加载模型权重
if opt.continue_train:
    utils.load_checkpoints_for_train(opt, generator, discriminator, netEMA)
else:
    generator.apply(utils.init_weights)
    discriminator.apply(utils.init_weights)
    netEMA = copy.deepcopy(generator) if not opt.no_EMA else None

# Optimizers
optimizerG = jt.optim.AdamW(generator.parameters(), lr=opt.lr_g, betas=(opt.beta1, opt.beta2))
optimizerD = jt.optim.AdamW(discriminator.parameters(), lr=opt.lr_d, betas=(opt.beta1, opt.beta2))

# 动态学习率
lr_shcedulerG = jt.lr_scheduler.CosineAnnealingLR(optimizer=optimizerG, T_max=len(dataloader) * 10, eta_min=0.00004)
lr_shcedulerD = jt.lr_scheduler.CosineAnnealingLR(optimizer=optimizerD, T_max=len(dataloader) * 10, eta_min=0.00001)

# 显示网络生成图像
visual_image_sample = utils.image_saver(opt, generator, netEMA, val_dataloader)

# 对continue_train的支持
already_started = False
start_epoch, start_iter = utils.get_start_iters(opt.loaded_latest_iter, len(dataloader))
print(f"start_iter:{start_iter},start_epoch:{start_epoch}")
cur_iter = start_epoch * len(dataloader) + start_iter

# 当前时间，用于计算训练剩余时间
prev_time = time.time()
for epoch in range(start_epoch, opt.num_epochs):
    for i, data_i in enumerate(dataloader):
        if not already_started and i < start_iter:
            print(f"skip {i} iter")
            continue
        already_started = True
        cur_iter = epoch * len(dataloader) + i
        image, label = utils.preprocess_input(opt, data_i)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        with jt.no_grad():
            fake = generator(label)
        output_D_fake = discriminator(fake)
        loss_D_fake = losses_computer.loss(output_D_fake, label, for_real=False)
        loss_D = loss_D_fake
        output_D_real = discriminator(image)
        loss_D_real = losses_computer.loss(output_D_real, label, for_real=True)
        loss_D += loss_D_real
        if not opt.no_labelmix:
            mixed_inp, mask = losses.generate_labelmix(label, fake, image, opt)
            output_D_mixed = discriminator(mixed_inp)
            loss_D_lm = opt.lambda_labelmix * losses_computer.loss_labelmix(mask, output_D_mixed,
                                                                            output_D_fake,
                                                                            output_D_real)
            loss_D += loss_D_lm
            loss_D.mean()
        optimizerD.step(loss_D)
        lr_shcedulerD.step()

        # ------------------
        #  Train Generators
        # ------------------
        fake = generator(label)
        output_D = discriminator(fake)
        loss_G = losses_computer.loss(input=output_D, label=label, for_real=True)
        # if opt.add_vgg_loss:
        #     loss_G_vgg = opt.lambda_vgg * VGG_loss(fake, image)
        #     loss_G += loss_G_vgg
        # else:
        loss_G_vgg = None
        optimizerG.step(loss_G)
        lr_shcedulerG.step()

        # jt.display_memory_info()
        if not opt.no_EMA:
            utils.update_EMA(netEMA, generator, cur_iter, dataloader, opt=opt)

        jt.sync_all()
        jt.gc()
        if jt.rank == 0:

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            iter_left = opt.num_epochs * len(dataloader) - cur_iter
            time_left = datetime.timedelta(seconds=iter_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            if cur_iter % opt.freq_print == 0:
                print("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f，VGG_Loss: %f][lrd:%f;lrg:%f] ETA: %s"
                      % (epoch, opt.num_epochs, i, len(dataloader), loss_D.numpy()[0],
                         loss_G.numpy()[0], loss_G_vgg.numpy()[0] if loss_G_vgg is not None else 0,
                         optimizerD.lr, optimizerG.lr, time_left))
            if cur_iter % opt.freq_save_ckpt == 0:
                utils.save_networks(opt, cur_iter, generator, discriminator, netEMA)
            if cur_iter % opt.freq_val == 0:
                visual_image_sample.visualize_batch(cur_iter)
                utils.save_networks(opt, cur_iter, generator, discriminator, netEMA, latest=True)
