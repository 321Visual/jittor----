| 第二届计图挑战赛开源模板

# Jittor赛题一(自然场景图像生成)
## 简介
本项目包含了第二届计图挑战赛热身赛（自然场景图像生成）的代码实现。本团队实现了OASIS网络的jittor版本，并在其基础上进行改动，使其可以适用于本赛题任务。参考的论文以及Pytorch代码实现如下：

+ Pytroch代码： [GitHub - boschresearch/OASIS: Official implementation of the paper "You Only Need Adversarial Supervision for Semantic Image Synthesis" (ICLR 2021)](https://github.com/boschresearch/OASIS)
+ 论文：[arxiv.org](https://arxiv.org/abs/2012.04781)

## 安装 
#### 运行环境
- ubuntu 18..04 LTS
- python >= 3.7
- jittor == 1.3.4.7

#### 安装依赖
执行以下命令安装 python 依赖
```shell
pip install -r requirements.txt
```

## 数据集

使用jittor大赛数据进行训练需要在大赛官网进行数据集的下载。

也可以使用公开数据集COCO-Stuff，Cityscapes 或者ADE20K数据集进行模型训练。参考https://github.com/NVlabs/SPADE项目数据集的具体配置

## 训练

+ #### 硬件配置

  我们在6张2080T显卡上训练了本项目模型，其中batchsize设置为6

运行以下命令进行模型的训练（首次运行注意替换脚本中数据集的对应位置

并通过直接在config.py代码文件的相应位置修改参数进行 **继续训练** 以及 训练 **其他参数**的指定：

```shell
./scripts/multi_gpu.sh
```

## 图像生成

训练好的模型文件已经包含在**checkpoints\competition\models**中，可以使用如下命令进行图像生成工作

```shell
./scripts/generate_image_gpu.sh
```

