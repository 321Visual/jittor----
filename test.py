import cv2
import config
import datasets
from utils import utils
from generator import OASIS_Generator
import os
import jittor as jt

if __name__ == '__main__':
    jt.cudnn.set_max_workspace_ratio(0.0)
    jt.flags.use_cuda = 1
    opt = config.read_arguments(train=False)
    opt.no_spectral_norm = True
    opt.which_iter = 1200
    opt.batch_size = 10
    opt.checkpoints_dir = './checkpoints'
    utils.fix_seed(opt.seed)

    val_dataloader = datasets.ImageDataset(opt, mode="test").set_attrs(
        batch_size=10,
        shuffle=False,
        num_workers=1,
    )

    model = OASIS_Generator(opt)

    which_iter = "latest" if opt.which_iter == -1 else opt.which_iter
    # path = os.path.join(opt.checkpoints_dir, opt.name, "models")
    # path = os.path.join(opt.output_path, "multi_gpu11", "saved_models/")
    path = os.path.join(opt.checkpoints_dir, opt.name, "models", str(which_iter))
    model.load(path + "_EMA.pkl")

    print("\n >>generate_imgs<< \n")
    # os.makedirs(f"{opt.output_path}/generated_imgs/img", exist_ok=True)
    os.makedirs(opt.output_path, exist_ok=True)
    # os.makedirs(f"{opt.output_path}/generated_imgs/color_label", exist_ok=True)
    completed = 0
    for i, data_i in enumerate(val_dataloader):

        label = data_i["label"]
        color_label = utils.Colorize(label.data)
        _, label_t = utils.preprocess_input(opt, data_i)
        print("生成中......")
        img = model(label_t)
        img = ((img + 1) / 2 * 255).numpy().astype('uint8')

        for idx in range(img.shape[0]):
            completed += 1
            img_name = data_i['name'][idx][:-4] + ".jpg"
            cv2.imwrite(f"{opt.output_path}/{os.path.split(img_name)[-1]}",
                        img[idx].transpose(1, 2, 0)[:, :, ::-1])
            # cv2.imwrite(f"{opt.output_path}/generated_imgs/color_label/{os.path.split(img_name)[-1]}",
            #             color_label[idx].transpose(1, 2, 0)[:, :, ::-1])
            print("已经完成:{}，| {}".format(img_name, completed))
