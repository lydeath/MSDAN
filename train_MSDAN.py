import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from DerainDataset import *
from utils import *
from torch.optim.lr_scheduler import MultiStepLR
from SSIM import SSIM
from networks import *


parser = argparse.ArgumentParser(description="MSDAN_train")
parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=[30,50,80], help="When to decay learning rate")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--save_path", type=str, default="logs/Rain100L/MSDAN", help='path to save models and log files')
parser.add_argument("--save_freq",type=int,default=1,help='save intermediate model')
parser.add_argument("--data_path",type=str, default="datasets/train/RainTrainL",help='path to training data')
# parser.add_argument("--data_path",type=str, default="datasets/train/RainTrainH",help='path to training data')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=3, help='number of recursive stages')
opt = parser.parse_args()

if opt.use_gpu:
    # gpu_id 默认为0
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

def main():

    print('开始运行')

    print('载入数据。。。/n')
    dataset_train = Dataset(data_path=opt.data_path)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batch_size, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))

    ############################################################################################################
    # print('开始')
    # f = h5py.File('E:/论文复现/MSDAN-master - 副本/datasets/train/Rain12600/train_target.h5', 'r')
    #
    # for root_name, g in f.items():
    #     print(root_name)
    #     for _, weights_dirs in g.attrs.items():
    #         for i in weights_dirs:
    #             name = root_name + "/" + str(i, encoding="utf-8")
    #             data = f[name]
    #             print(data.value)
    #
    # print('结束')
    ############################################################################################################

    #################################  建立模型  #####################################################################
    # Build model
    model = MSDAN(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)
    print('打印网络')
    print_network(model)

    # 损失函数
    criterion = SSIM()

    # Move to GPU
    if opt.use_gpu:
        model = model.cuda()
        criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.2)  # learning rates

    # record training  记录训练
    writer = SummaryWriter(opt.save_path)  # save_path = logs/MSDAN_test

    # 载入最新的模型
    initial_epoch = findLastCheckpoint(save_dir=opt.save_path)
    if initial_epoch > 0:
        print('剩余的循环  ：resuming by loading epoch %d' % initial_epoch)
        # 加载权重
        model.load_state_dict(torch.load(os.path.join(opt.save_path, 'net_epoch%d.pth' % initial_epoch)))

    # 开始训练
    step = 0
    # 从之前加载的最后一个epoch开始运行
    for epoch in range(initial_epoch, opt.epochs):
        scheduler.step(epoch)
        # 遍历参数
        for param_group in optimizer.param_groups:
            print('当前学习率 learning rate %f' % param_group["lr"])

        # epoch training start 开始训练
        for i, (input_train, target_train) in enumerate(loader_train, 0):
            # print(input_train.shape)  # [16, 3, 100, 100]
            # print(target_train.shape)  # [16, 3, 100, 100]
            ###############################################################################################
            # 如果模型中有BN层(BatchNormalization）和 Dropout，需要在训练时添加model.train()。model.train()
            # 是保证BN层能够用到每一批数据的均值和方差。对于Dropout，model.train()是随机取一部分网络连接来训练更新参数。
            # 不启用BatchNormalization和Dropout。
            # 如果模型中有BN层(BatchNormalization）和Dropout，在测试时添加model.eval()。model.eval()
            # 是保证BN层能够用全部训练数据的均值和方差，即测试过程中要保证BN层的均值和方差不变。对于Dropout，model.eval()
            # 是利用到了所有网络连接，即不进行随机舍弃神经元。
            ###############################################################################################
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            input_train, target_train = Variable(input_train), Variable(target_train)

            if opt.use_gpu:
                input_train, target_train = input_train.cuda(), target_train.cuda()


#############################################################################################
            # 开始训练
            out_train, _ = model(input_train)
            # target_train 对比图片
            # out_train 输出图片
            pixel_metric = criterion(target_train, out_train)
            loss = -pixel_metric

            loss.backward()
            optimizer.step()


#############################################################################################
            # 训练曲线
            model.eval()
            out_train, _ = model(input_train)
            out_train = torch.clamp(out_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, target_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f, pixel_metric: %.4f, PSNR: %.4f" %
                  (epoch+1, i+1, len(loader_train), loss.item(), pixel_metric.item(), psnr_train))

            if step % 10 ==0:

                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1


# log the images
        # 记录图像
        model.eval()
        out_train, _ = model(input_train)
        out_train = torch.clamp(out_train, 0., 1.)
        im_target = utils.make_grid(target_train.data, nrow=8, normalize=True, scale_each=True)
        im_input = utils.make_grid(input_train.data, nrow=8, normalize=True, scale_each=True)
        im_derain = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', im_target, epoch+1)
        writer.add_image('rainy image', im_input, epoch+1)
        writer.add_image('deraining image', im_derain, epoch+1)
        # writer.add_image('clean image', target_train.data, epoch+1)
        # writer.add_image('rainy image', input_train.data, epoch+1)
        # writer.add_image('deraining image', out_train.data, epoch+1)

        # save model  保存最后的pth
        torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_latest.pth'))
        if epoch % opt.save_freq == 0:  # 保存频率
            torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_epoch%d.pth' % (epoch+1)))


if __name__ == "__main__":
    if opt.preprocess:
        if opt.data_path.find('RainTrain1000') != -1:
            print('RainTrain1000')
            # strid 步长80
            prepare_data_RainTrain1000(data_path=opt.data_path, patch_size=100, stride=80)
        else:
            print('unkown datasets: please define prepare data function in DerainDataset.py')


main()

