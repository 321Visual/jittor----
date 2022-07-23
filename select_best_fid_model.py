from utils.fid_scores import fid_jittor
import datasets
import jittor as jt
import config
from utils import utils
from generator import OASIS_Generator
import os

if __name__ == '__main__':
    jt.cudnn.set_max_workspace_ratio(0.0)
    jt.flags.use_cuda = 1
    opt = config.read_arguments(train=False)
    opt.no_spectral_norm = True
    utils.fix_seed(opt.seed)
    val_dataloader = datasets.ImageDataset(opt, mode="val").set_attrs(
        batch_size=10,
        shuffle=False,
        num_workers=1,
    )
    fid_computer = fid_jittor(opt=opt, dataloader_val=val_dataloader)
    models_dir = os.path.join(opt.checkpoints_dir, opt.name, "models")

    generator_files = []
    EMA_files = []

    for file in sorted(os.listdir(models_dir)):
        if str(file).endswith('EMA.pkl'):
            EMA_files.append(os.path.join(models_dir, str(file)))
        if str(file).endswith('G.pkl'):
            generator_files.append(os.path.join(models_dir, str(file)))
    generator_files.sort()
    EMA_files.sort()

    generator = OASIS_Generator(opt)
    EMA = OASIS_Generator(opt)

    best_name = ''

    for i, (generator_path, ema_path) in enumerate(zip(generator_files, EMA_files)):
        generator.load(generator_path)
        EMA.load(ema_path)
        is_best = fid_computer.update(generator, EMA, i)
        if is_best:
            best_name = ema_path
            print(best_name)
    print(best_name)
