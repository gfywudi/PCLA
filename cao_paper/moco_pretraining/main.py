#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Jiewei Lai
from sklearn import preprocessing
from torch.autograd import Variable


from tqdm import tqdm
import os
import scipy
import random
import shutil
import argparse
import builtins

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.parallel
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import sys
sys.path.append("/home/guofengyi/code/mamba-main")
import MSDNN
from RSTAnet import ResNet1D, ResNet1DBlock
# from mamba_small import Model
import builder
from augmentation import ECGFrequencyDropOut, ECGCycleMask, ECGCropResize, ECGChannelMask
from cao_routine_training.single_lead_config import get_train_config
from cao_routine_training.load_data_single_lead import load_data
from scipy import signal
import sklearn.metrics as M

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '4348'

#=======================================================================================================================
# Read ECG and return a signal with size [n, 12]
def ecg_read(path):
    
    data = open(path, 'rb').read()
    ecgdata = []
    for i in range(0, len(data), 2):
        a = data[i]
        b = data[i+1] * 256
        if (a+b) >= 32768:
            ecgdata.append(a + b - 65536)
        else:
            ecgdata.append(a + b)
    
    ecg_len = len(ecgdata)
    ecgdata = np.array(ecgdata[:(ecg_len - ecg_len % 12)])
    ecgdata[np.isnan(ecgdata)] = 0
    data = ecgdata.reshape(-1, 12)
    
    return data

# High pass filter, 5th order, 0.5Hz
def ecg_HP(data):

    low = 0.5 / 250
    b, a = signal.butter(5, low, btype='highpass')
    data_HP = signal.filtfilt(b, a, data, axis=0)

    return data_HP

class ThreeTransform:
    """Take three random crops of one ECG as the query, key and another three views."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x).astype(np.float32)
        k = self.base_transform(x).astype(np.float32)
        t1 = self.base_transform(x).astype(np.float32)
        t2 = self.base_transform(x).astype(np.float32)
        t3 = self.base_transform(x).astype(np.float32)
        t = [t1, t2, t3]
        return q, k, t

class ECGDataSet(torch.utils.data.Dataset):

    def __init__(self, ecg_dirs, transform):
        self.ecg_dirs = ecg_dirs
        self.transform = transform

    def __getitem__(self, index):
        data = self.ecg_dirs[index,:,:]
        # path = self.ecg_dirs[index]
        # if not path.endswith('.bin'):
        #     path += '.bin'
        # data = ecg_HP(ecg_read(path)).T
        #
        # if data.shape[-1] < 5000:
        #     data_resize = np.empty([12, 5000])
        #     x = np.linspace(0, data.shape[-1] - 1, data.shape[-1])
        #     xnew = np.linspace(0, data.shape[-1] - 1, 5000)
        #     for i in range(data.shape[0]):
        #         f = scipy.interpolate.interp1d(x, data[i], kind='cubic')
        #         data_resize[i] = f(xnew)
        #     data = data_resize
        #
        # if data.shape[-1] > 5000:
        #     start = random.randint(0, data.shape[-1] - 5000)
        #     data = data[:, start:start + 5000]
        if self.transform is not None:
           data = self.transform(data)

        return data

    def __len__(self):
        return len(self.ecg_dirs)

# def main_worker(gpu, ngpus_per_node, args):
def main_worker(gpu,args):

    args.gpu = gpu

    # args.nr = ngpus_per_node

    # Only print the contents of GPU0
    if args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    args.rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank)

    # Create model
    print("Creating model: MSDNN.")

    def base_encoder(num_classes=1000):
        # 假设你要创建一个 ResNet1D 模型，并指定输出维度
        model = ResNet1D(ResNet1DBlock, [2, 2, 2, 2, 2, 2, 2], num_classes=num_classes)
        return model
    # Model = ResNet1D(ResNet1DBlock, [2, 2, 2, 2, 2, 2, 2], num_classes=args.moco_dim)
    # model = builder.MoCo(MSDNN.MSDNN, args.moco_dim, args.moco_k, args.moco_m, args.moco_t)
    model = builder.MoCo(base_encoder, args.moco_dim, args.moco_k, args.moco_m, args.moco_t)

    try:
        state_dict = torch.load(os.path.join(args.modelpath, 'ECGMoCo_MSCNN_mamba.pth'), map_location="cpu")
        state_dict['queue_ptr'][0] = 0
        model.load_state_dict(state_dict)
        print("Pre-trained MoCo loaded.")
    except:
        print("Pre-trained network initialized.")


    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    args.batch_size = int(args.batch_size / args.gpus)
    args.workers = int((args.workers + args.gpus - 1) / args.gpus)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.Adam(model.parameters(), args.weight_decay)

    cudnn.benchmark = True

    augmentation = [transforms.RandomApply([ECGFrequencyDropOut()]),
                    transforms.RandomApply([ECGCycleMask()]),
                    transforms.RandomApply([ECGCropResize()])]
                    # transforms.RandomApply([ECGChannelMask()])]

    # ecgpath = '/home/laijiewei/ECGProject/ECG/样本库数据/'
    # label_info = pd.read_excel(os.path.join(ecgpath, '正常数据.xlsx'))
    # ecg_dir1 = label_info['sample_data'].values
    # ecg_dir1 = [os.path.join(ecgpath, dir) for dir in ecg_dir1]
    #
    # np.random.seed(8)
    # unlabel_info = pd.read_excel("/home/laijiewei/ECGProject/ECG/无标签数据/正常数据.xlsx")
    # ecg_dir2 = unlabel_info['sample'].values
    # ecg_dir2 = np.random.choice(ecg_dir2, int(len(ecg_dir2) * args.ratio), replace=False).tolist()

    # ecg_dirs = ecg_dir1 + ecg_dir2
    # np.random.shuffle(ecg_dirs)    configs = get_train_config()
    configs = get_train_config()
    data_iter = load_data(configs)
    X_train = data_iter[0]
    y_train = data_iter[1]
    if np.isnan(X_train).any():
        print("数据中存在 NaN 值，进行处理...")
        # 使用均值填充 NaN 值
        # 注意：为了避免在计算均值时引入 NaN，先将 NaN 替换为 0
        X_train = np.nan_to_num(X_train, nan=0)  # 将 NaN 替换为 0
    lead = [1]
    X_train =np.swapaxes(X_train[:, :, lead],1,2)

    train_dataset = ECGDataSet(X_train, transform=ThreeTransform(transforms.Compose(augmentation)))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    if args.reuse:
        result = pd.read_excel(os.path.join(args.modelpath, 'result_Resmamba.xlsx'))
        start = result.__len__()
    else:
        result = creat_logs()
        start = 0

    for epoch in range(start, args.epochs):

        train_sampler.set_epoch(epoch)

        # train for one epoch
        loss_c, loss_d, acc = train(train_loader, model, criterion, optimizer, epoch, args, result)
        save_logs(args, result, model, epoch, loss_c, loss_d, acc)
        # plot_result(args, result)

def dimcon(x):
    t = []
    x_pin = torch.tensor(t)
    for i in range(12):
        x_er = x[:, :, i]
        x_era = preprocessing.scale(x_er)#z分数归一化
        x_era = torch.Tensor(x_era)
        x_era = Variable(x_era)
        x_era = x_era.unsqueeze(1)  # 将x的维度进行扩展
        x_pin = torch.cat((x_pin, x_era), dim=1, out=None)
    return x_pin







def creat_logs():

    result = pd.DataFrame(columns=['loss_c', 'loss_d', 'acc', 'epoch', 'best_acc'])
    result.loc[0, 'best_acc'] = None

    return result

def save_logs(args, result, model, epoch, loss_c, loss_d, acc):

    result.loc[epoch, 'loss_c'] = loss_c
    result.loc[epoch, 'loss_d'] = loss_d
    result.loc[epoch, 'acc'] = acc

    if result.loc[0, 'best_acc'] == None:
        print("The best_acc of the model is {:.2%}.\n".format(acc))
        result.loc[0, 'epoch'] = epoch
        result.loc[0, 'best_acc'] = acc
        torch.save(model.module.state_dict(), os.path.join(args.modelpath, 'mamba_resnet.pth'))
        while True:
            try:
                torch.load(os.path.join(args.modelpath, 'mamba_resnet.pth'), map_location="cpu")
                break
            except:
                torch.save(model.module.state_dict(), os.path.join(args.modelpath, 'mamba_resnet.pth'))
                print("Save MoCo again.")
    elif result.loc[0, 'best_acc'] < acc:
        print("The best_acc of the model is improved from {:.2%} to {:.2%}.\n".format(result.loc[0, 'best_acc'], acc))
        result.loc[0, 'epoch'] = epoch
        result.loc[0, 'best_acc'] = acc
        torch.save(model.module.state_dict(), os.path.join(args.modelpath, 'mamba_resnet.pth'))
        while True:
            try:
                torch.load(os.path.join(args.modelpath, 'mamba_resnet.pth'), map_location="cpu")
                break
            except:
                torch.save(model.module.state_dict(), os.path.join(args.modelpath, 'mamba_resnet.pth'))
                print("Save MoCo again.")
    else:
        print("The best_acc of the model didn't improved from {:.2%}.\n".format(result.loc[0, 'best_acc']))

    result.to_excel(os.path.join(args.modelpath, 'result_mamba_resnet.xlsx'))

def plot_result(args, result):

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.semilogy()
    ax1.plot(result['loss_c'], 'orange', label='loss_contrastive')
    ax1.plot(result['loss_d'], 'gold', label='loss_distribute')
    ax1.legend(loc=2)
    ax2.plot(result['acc'], 'purple', label='acc')
    ax2.legend(loc=3)
    ax2.scatter(result.loc[0, 'epoch'], result.loc[0, 'best_acc'], linewidth=2)

    ax1.set_ylabel("Loss")
    ax2.set_ylabel("ACC")

    plt.grid()
    plt.title('ACC: {:.2%}'.format(result.loc[0, 'best_acc']))

    plt.savefig(os.path.join(args.modelpath, 'result'))
    plt.clf()
    plt.close()

def train(train_loader, model, criterion, optimizer, epoch, args, result):

    # switch to train mode
    model.train()
    samples = len(train_loader.dataset)
    loss_c = list()
    loss_d = list()
    acc = list()
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    #
    for batch_idx, (batch_q, batch_k, batch_s) in loop:
    # for batch_idx, (batch_q, batch_k, batch_s) in enumerate(train_loader):
        # print("batch:", batch_idx)
        if args.gpu is not None:
            batch_q = batch_q.cuda(args.gpu, non_blocking=True)
            batch_k = batch_k.cuda(args.gpu, non_blocking=True)
            batch_s = [s.cuda(args.gpu, non_blocking=True) for s in batch_s]

        # compute output
        output, target, loss_contrastive, loss_distribute = model(batch_q, batch_k, batch_s)
        batch_loss = loss_contrastive + loss_distribute
        batch_acc = M.accuracy_score(target.data.cpu().numpy(), torch.argmax(output, 1).data.cpu().numpy())

        loss_c.append(loss_contrastive.item())
        loss_d.append(loss_distribute.item())
        acc.append(batch_acc)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        loop.set_description(f'Epoch [{epoch}]')
        loop.set_postfix(loss_c=loss_contrastive.item(), loss_d= loss_distribute.item(), acc=batch_acc)

    # print('Train Epoch: {:.2f}|{} {} LossC: {:.4f}  LossD: {:.4f}  ACC: {:.2%}'.format(
    #         epoch + batch_idx/len(train_loader), args.epochs, samples, loss_contrastive.item(),
    #         loss_distribute.item(), batch_acc))

    loss_c_mean = np.mean(loss_c)
    loss_d_mean = np.mean(loss_d)
    acc_mean = np.mean(acc)

    print('\nTrain Epoch: {:.2f}|{} {} LossC: {:.4f}  LossD: {:.4f}  ACC: {:.2%}\n'.format(
          epoch, args.epochs, samples, loss_c_mean, loss_d_mean, acc_mean))

    return loss_c_mean, loss_d_mean, acc_mean
#======================================================================================================================


if __name__ == '__main__':
    # Set path
    rootpath = os.path.dirname(os.path.abspath('..'))
    datapath = os.path.join(rootpath, 'data')
    modelpath = os.path.join(datapath, 'ECGMoCo_Resmamba_PTBXL')

    # 设置超参数
    parser = argparse.ArgumentParser(description='PyTorch ECG Training')
    parser.add_argument('--datapath', default=None, type=str,
                        help='path to dataset')
    parser.add_argument('--modelpath', default=None, type=str,
                        help='path to model')
    parser.add_argument('-j', '--workers', default=2, type=int,
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')

    parser.add_argument('-n', '--nodes', default=1, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=torch.cuda.device_count(), type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    # parser.add_argument('--gpus', default=3, type=int,
    #                     help='GPU number to use.')

    # moco specific configs:
    parser.add_argument('--moco-dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--moco-k', default=2560, type=int,
                        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('--moco-m', default=0.9, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco-t', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')

    parser.add_argument('--reuse', default=False, type=bool,
                        help='Continue training the network')

    args = parser.parse_args()

    # if not args.reuse:
    #     if os.path.exists(modelpath):
    #         shutil.rmtree(modelpath)
    #     os.makedirs(modelpath, exist_ok=True)

    args.modelpath = "/homeb/guofengyi/result/cao_paper/few_shot_result_txt"

    args.world_size = args.gpus * args.nodes

    # main(args.gpu,args)
    # main_worker(args.gpus, args)
    mp.spawn(main_worker, nprocs=args.gpus, args=(args,))
    # mp.spawn(main_worker, nprocs=args.gpus, args=(args.gpus, args))
    # mp.spawn(main_worker, nprocs=args.gpus, args=(args.gpus, args))
    # mp.spawn(main, nprocs=args.gpus, args=(args.gpus, args))