import pickle
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from single_lead_config import get_train_config

# from Network import LogisticRegressionModel
from torch.utils.data import Dataset, DataLoader

from meta_test_loader_data import get_test_dataset
from meta_train_loader_data import get_train_dataset

import os
import sys
import time
import torch.backends.cudnn
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from prototypicalNetwork import PrototypicalNetwork
from cao_routine_training.RSTAnet import ResNet1D,ResNet1DBlock
# from cao_routine_training.RSTAnet_xiaorong_resblock_number import ResNet1D,ResNet1DBlock
# from cao_routine_training.RSTA_mamba_no_SElayer import ResNet1D,ResNet1DBlock


import torch.backends.cudnn as cudnn
from tqdm import tqdm
# from util import adjust_learning_rate, accuracy, AverageMeter
# import wandb
# from torchmetrics import MetricCollection
# from torchmetrics.classification import Accuracy, Precision, Recall, AUROC, F1Score
# 除去 precision的报错
# import warnings
# warnings.filterwarnings("ignore")

from sklearn.preprocessing import normalize


def main(lr_rate, project_name, configs):
    configs.final_lr = lr_rate
    num_class = configs.n_way

    best_model = None

    data_size = int(5000 / configs.data_dim)

    # 用于存储每个折叠的训练数据的损失和指标
    train_loss_per_fold = []
    train_metrics_per_fold = []

    "da  single_ecg_cnn_transfomer   _4class   lr_{configs.final_lr}_bz_{configs.batch_size}_{project_name}"
    # wandb.init(project=f"single_ecg_protonet_few_shot",
    #         config=configs,
    #         save_code=True,
    #         name=f"{configs.n_way}_way_{configs.k_shot}_shot_lr_{configs.final_lr}_bz_{configs.batch_size}_{project_name}")

    np.set_printoptions(linewidth=400, precision=4)

    '''set device'''
    os.environ["CUDA_VISIBLE_DEVICES"] = configs.device

    # '''load data'''
    # data_iter = load_data(configs)
    # print('=' * 20, 'load data over', '=' * 20)

    # # 将数据分割成训练集和剩余集（包括验证集和测试集）
    # X_train, X_test, y_train, y_test = train_test_split(data_iter[0], data_iter[1], test_size=0.2, random_state=42)
    # print(np.array(X_train).shape)

    # kf = KFold(n_splits=2, shuffle=True, random_state=42)

    print('=' * 50 + f'Start Training  {num_class} class' + '=' * 50)

    # ----------------------------------------------------在整个训练集上训练最终的模型----------------------------------------------------
    # model = ECGFeatureExtractor(input_channels=1, cnn_output_channels=64, model_dim=128, num_heads=2, num_layers=2)
    ecg_network = ResNet1D(ResNet1DBlock, [2, 2, 2, 2, 2, 2, 2], num_classes=num_class)
    # mdfile1 = '/homeb/guofengyi/result/cao_paper/few_shot_result_txt/moco_4mamba_MSDNN_PTBXL.pth'
    # if configs.pre_train:
    #     if os.path.exists(mdfile1):
    #         print("pre_train load mdfile...")
    #         # 加载预训练模型的状态字典
    #         pretrained_state_dict = torch.load(mdfile1)
    #         # 获取当前模型的状态字典
    #         model_state_dict = ecg_network.state_dict()
    #         # 从预训练状态字典中删除最后一层的权重和偏置
    #         pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict and "fc" not in k}
    #         # 更新当前模型的状态字典
    #         model_state_dict.update(pretrained_state_dict)
    #         # 加载更新后的状态字典到模型中
    #         ecg_network.load_state_dict(model_state_dict)
    # 现在模型除了最后一层之外的其他层都加载了预训练的权重
    model = PrototypicalNetwork(encoder=ecg_network, n_ways=configs.n_way, n_shots=configs.k_shot,
                                n_querys=configs.n_query,device="cuda:0")
    # 计算模型参数的总大小（以字节为单位）
    total_params_size = sum(p.numel() * p.element_size() for p in model.parameters())

    # 将字节转换为更易读的格式，如MB或GB
    total_params_size_mb = total_params_size / (1024 ** 2)  # 转换为MB
    print(f"模型大小: {total_params_size_mb:.2f} MB")

    # resnet = ResNet1D(ResNet1DBlock, [2, 2, 2, 2], num_classes=2)
    # transformer = TransformerEncoder(input_dim=512, num_heads=8, dim_feedforward=256, num_layers=2)
    # model = ResNet1DTransformer(resnet, transformer, num_classes=num_class)

    best_recall_scores = 0.0
    best_precision_scores = 0.0
    best_f1_scores = 0.0
    best_roc_auc_scores = 0.0
    best_acc_scores = 0.0

    best_final_model = None
    final_criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        final_criterion = final_criterion.cuda()
        cudnn.benchmark = True
    final_optimizer = torch.optim.AdamW(model.parameters(), lr=configs.final_lr, weight_decay=1e-6)
    # final_optimizer = torch.optim.Adam(model.parameters(), lr=configs.final_lr)
    # final_optimizer = torch.optim.SGD(model.parameters(), lr=configs.final_lr, momentum=0.0003, weight_decay=1e-6)
    # 定义学习率调度器，这里使用StepLR调度器，每个epoch降低学习率
    # scheduler = torch.optim.lr_scheduler.StepLR(final_optimizer, step_size=5, gamma=0.5)  # 每个epoch降低学习率

    # # 创建数据加载器
    # train_dataset, val_dataset, test_dataset= get_train_dataset(configs)
    train_dataset = get_train_dataset(configs)
    configs.dataset_type_train = 'PTB'
    test_dataset = get_train_dataset(configs)

    final_test_loader = DataLoader(test_dataset, batch_size=configs.train_batch_size, shuffle=True, num_workers=0)
    final_train_loader = DataLoader(train_dataset, batch_size=configs.test_batch_size, shuffle=True, num_workers=0)

    # final_train_metrics = MetricCollection({
    #         'acc': Accuracy(task="multiclass", num_classes= num_class),
    #         'prec': Precision(task="multiclass", num_classes= num_class, average='weighted'),
    #         'rec': Recall(task="multiclass", num_classes= num_class, average='weighted'),
    #         'f1': F1Score(task="multiclass", num_classes= num_class, average='weighted'),
    #         'roc': AUROC(task="multiclass", num_classes= num_class, average='weighted')
    #     })

    # final_test_metrics = MetricCollection({
    #     'acc': Accuracy(task="multiclass", num_classes= num_class),
    #     'prec': Precision(task="multiclass", num_classes= num_class, average='weighted'),
    #     'rec': Recall(task="multiclass", num_classes= num_class, average='weighted'),
    #     'f1': F1Score(task="multiclass", num_classes= num_class, average='weighted'),
    #     'roc': AUROC(task="multiclass", num_classes= num_class, average='weighted')
    # })


    best_acc = 0.0
    # 在整个训练集上进行模型训练
    for epoch in range(configs.epoch):
        # scheduler.step()
        model.train()
        correct_all = 0
        total = 0
        loop = tqdm(enumerate(final_train_loader), total=len(final_train_loader))
        for idx, batch in loop:
        # for idx, batch in enumerate(final_train_loader):
            final_optimizer.zero_grad()
            support_x = batch[
                0].cuda()  # [batch_size, n_way*(k_shot+k_query), c , h , w] torch.Size([11, 5, 1, 5000, 1])
            support_y = batch[1].cuda()
            query_x = batch[2].cuda()
            query_y = batch[3].cuda()

            # print("support_y",support_y)
            bh, n_way, n_shot, lg, c = support_x.size()
            set1 = n_way * n_shot
            set2 = query_x.size(1) * query_x.size(2)

            support_set = support_x.view(bh * n_way * n_shot, c, lg).float()
            support_labels = support_y.view(bh * n_way * n_shot)
            # print("support_y", support_y.shape)
            query_set = query_x.view(bh * set2, c, lg).float()
            query_labels = query_y.view(bh * set2)

            # 前向传播和计算损失
            # outputs = model(support_set, support_labels, query_set)
            # loss = nn.CrossEntropyLoss()(outputs, query_labels)
            # 前向传播
            similarities,similarities_enhance,clloss = model(support_set, support_labels, query_set)

        # 计算损失
            # print("similarities",similarities.shape)
            # print("query_labels", query_labels.shape)
            loss_1 = model.loss(similarities, query_labels)
            loss_2 = model.loss(similarities_enhance, query_labels)
            # loss = loss_1+loss_2+clloss
            loss = loss_1 + loss_2
            # loss = loss_1

            loss.backward()
            final_optimizer.step()

            # print("outputs", outputs.size(),outputs)
            # 应用 softmax 函数将输出转换为概率分布
            # probabilities = F.softmax(outputs, dim=1)
            # 计算准确率
            # final_batch_metrics = final_train_metrics.cuda().forward(dists, query_labels)
            # print("similarities",similarities.shape)
            _, preds = torch.max(similarities, dim=2)
            # print(preds)
            correct = (preds == query_labels.view(-1)).sum().item()
            correct_all += correct
            total += query_labels.size(0)
            # print("preds", preds.size(),preds)

            # print(f"train Epoch: [{epoch}][{idx}/{len(final_train_loader)}]\t"
            #         f"Accuracy: {final_batch_metrics['acc']:.4f}\t"
            #         f"Loss: {loss.item():.4f}\t"
            #         f"f1_score: {final_batch_metrics['f1']:.4f}\t"
            #         f"roc_auc_score: {final_batch_metrics['roc']:.4f}\t"
            #         f"recall_score: {final_batch_metrics['rec']:.4f}\t"
            #         f"precision_score: {final_batch_metrics['prec']:.4f}")

            loop.set_description(f"train Epoch: [{epoch}][{idx}/{len(final_train_loader)}]\t")
            # loop.set_postfix(loss=loss / (i + 1), acc=float(right) / float(BATCH_SIZE * step + len(batch_x)))
            loop.set_postfix(Accuracy=correct / query_labels.size(0),
                         Loss=loss.item())
            # print(f"train Epoch: [{epoch}][{idx}/{len(final_train_loader)}]\t"
            #       f"Accuracy: {correct / query_labels.size(0):.4f}\t"
            #       f"loss: {loss.item():.4f}\t")
        # 在每个epoch结束时打印准确率
        accuracy = correct_all / total
        print(f'Epoch {epoch + 1}/{configs.epoch}, Accuracy: {accuracy}')
        if epoch % 1 == 0:
            test_accuracies = []  # 用于存储每个批次的准确率
            test_losses = []  # 用于存储每个批次的损失
            test_f1_scores = []  # 用于存储每个批次的f1分数
            test_roc_auc_scores = []
            test_recall_scores = []
            test_precision_scores = []
            predict_list = []
            label_list = []
            avg_test_accuracy = 0.0
            avg_test_loss = 0.0

            test_correct_all = 0
            test_total = 0
            model.eval()
            with torch.no_grad():
                # loop = tqdm(enumerate(final_test_loader), total=len(final_test_loader))
                # for idx, batch in loop:
                for idx, batch in enumerate(final_test_loader):
                    support_x = batch[
                        0].cuda()  # [batch_size, n_way*(k_shot+k_query), c , h , w] torch.Size([11, 5, 1, 5000, 1])
                    support_y = batch[1].cuda()
                    query_x = batch[2].cuda()
                    query_y = batch[3].cuda()

                    bh, n_way, n_shot, lg, c = support_x.size()
                    set1 = n_way * n_shot
                    set2 = query_x.size(1) * query_x.size(2)

                    support_set = support_x.view(bh * n_way * n_shot, c, lg).float()
                    # print("support_set", support_set.size())
                    support_labels = support_y.view(bh * n_way * n_shot)
                    query_set = query_x.view(bh * set2, c, lg).float()
                    query_labels = query_y.view(bh * set2)

                    # 前向传播和计算损失
                    # 前向传播
                    similarities, similarities_enhance, clloss = model(support_set, support_labels, query_set)

                    # 计算损失
                    # print("similarities",similarities.shape)
                    # print("query_labels", query_labels.shape)
                    # similarities = model(support_set, support_labels, query_set)
                    # 计算损失
                    loss = model.loss(similarities, query_labels)
                    # print("outputs", outputs.size(),outputs)
                    # 应用 softmax 函数将输出转换为概率分布
                    # probabilities = F.softmax(outputs, dim=1)
                    _, preds = torch.max(similarities, dim=2)
                    predict_list.append(preds)
                    label_list.append(query_labels)
                    test_correct = (preds == query_labels).sum().item()
                    test_correct_all += test_correct
                    test_total += query_labels.size(0)
                    # loop.set_description(f"train Epoch: [{epoch}][{idx}/{len(final_train_loader)}]\t")
                    # # loop.set_postfix(loss=loss / (i + 1), acc=float(right) / float(BATCH_SIZE * step + len(batch_x)))
                    # loop.set_postfix(Accuracy=test_correct / query_labels.size(0))
                    # print(f"test Epoch: [{epoch}][{idx}/{len(final_test_loader)}]\t"
                    #       f"Accuracy: {test_correct / query_labels.size(0):.4f}\t")

                    # final_batch_metrics = final_train_metrics.cuda().forward(preds, query_labels)

                    # print(f"test Epoch: [{epoch}][{idx}/{len(final_test_loader)}]\t"
                    #     f"Accuracy: {test_metrics['acc']:.4f}\t"
                    #     f"Loss: {loss.item():.4f}\t"
                    #     f"f1_score: {test_metrics['f1']:.4f}\t"
                    #     f"roc_auc_score: {test_metrics['roc']:.4f}\t"
                    #     f"recall_score: {test_metrics['rec']:.4f}\t"
                    #     f"precision_score: {test_metrics['prec']:.4f}")

                    # test_accuracies.append(test_metrics['acc'])
                    # test_f1_scores.append(test_metrics['f1'])
                    # test_roc_auc_scores.append(test_metrics['roc'])
                    # test_recall_scores.append(test_metrics['rec'])
                    # test_precision_scores.append(test_metrics['prec'])
                    # test_losses.append(loss.item())

                test_accuracy = test_correct_all / test_total

                print(
                    f'Epoch {epoch + 1}/{configs.epoch}, test Accuracy: {test_accuracy:.4f}, test loss: {loss.item():.4f}:')
                if best_acc < test_accuracy:
                    best_acc = test_accuracy
                    torch.save(model.state_dict(),
                               '/homeb/guofengyi/result/cao_paper/fewshot/model/'+f"{configs.dataset_type_train}_to{configs.dataset_type_test}"+model_name+f'_lr{configs.final_lr}_bz_{configs.batch_size}_{configs.n_way}way_{configs.k_shot}_shot_{configs.n_query}_query.pth')
                    best_final_model = model
                    # print(avg_test_accuracy)
                # # 计算平均准确率
                # results = {
                #     'acc': sum(test_accuracies) / len(test_accuracies),  # 计算验证集上的平均准确率
                #     'f1': sum(test_f1_scores) / len(test_f1_scores),
                #     'roc': sum(test_roc_auc_scores) / len(test_roc_auc_scores),
                #     'rec': sum(test_recall_scores) / len(test_recall_scores),
                #     'prec': sum(test_precision_scores) / len(test_precision_scores),
                #     'loss': sum(test_losses) / len(test_losses)
                # }

                # 使用WandB记录验证集上的评估结果
                # wandb.log({f"train acc ": results['acc'],
                #             f"train loss ": results['loss'],
                #             f"train f1_score": results['f1'],
                #             f"train roc_auc_score": results['roc'],
                #             f"train recall_score": results['rec'],
                #             f"train precision_score": results['prec'],
                #             f"epoch" : epoch})
                # wandb.log({f"train acc ": test_accuracy,
                #            f"train loss ": loss.item(),
                #            f"epoch": epoch})

                # print(f"test Epoch: [{epoch}]\t"
                #         f"Accuracy: {results['acc']:.4f}\t"
                #         f"Loss: {results['loss']:.4f}\t"
                #         f"f1_score: {results['f1']:.4f}\t"
                #         f"roc_auc_score: {results['roc']:.4f}\t"
                #         f"recall_score: {results['rec']:.4f}\t"
                #         f"precision_score: {results['prec']:.4f}")

                # if best_acc_scores < results['acc']:
                #     best_acc_scores = results['acc']
                #     # 保存模型的状态字典到指定文件路径
                #     torch.save(model.state_dict(), f'model/1_{project_name}_lr{configs.final_lr}_bz_{configs.batch_size}_{configs.num_class}class.pth')
                #     best_final_model = model
                #     print(avg_test_accuracy)
                # if best_recall_scores < results['rec']:
                #     best_recall_scores = results['rec']
                # if best_precision_scores < results['prec']:
                #     best_precision_scores = results['prec']
                # if best_f1_scores < results['f1']:
                #     best_f1_scores = results['f1']
                # if best_roc_auc_scores < results['roc']:
                #     best_roc_auc_scores = results['roc']

    print(f"best model: {best_final_model}")
    print(f"Test Score : {best_acc}")

    # 结束WandB会话
    # wandb.finish()

    # best_record = [best_acc_scores, best_recall_scores, best_precision_scores, best_f1_scores, best_roc_auc_scores]
    return best_acc


if __name__ == '__main__':
    # 初始化
    configs = get_train_config()
    wayshot_list = [[3,5]]
    configs.device = '1'
    configs.dataset_type_train = 'PTB'
    configs.dataset_type_test = 'chapman'
    model_name = "RSTAnet_mamba_7_class_transform_contrast"
    for way_shot in wayshot_list:
        configs.n_way = way_shot[0]
        configs.k_shot = way_shot[1]
        best_Score_record = []
        # lr_rates = [0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.000005, 0.0000001, 0.0000005]  # 不同的学习率值
        lr_rates = [0.01,
                    0.05, 0.001, 0.005, 0.0001, 0.0005]
        # lr_rates = [0.00000001, 0.00000005]
        # lr_rates = [0.00000005]
        # lr_rates = [0.0005, 0.00001, 0.00005 ,0.000001, 0.000005, 0.0000001, 0.0000005]
        # lr_rates = [0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003 ,0.0002, 0.0001]
        # lr_rates = [0.00009,0.000001, 0.000002, 0.000003]
        # SGD Adam AdamW weight_decay=1e-7 "resnet1d+attention 1-16-128 "  resnet1d+attention_layer_1-64-512
        # project_name = "resnet1d+attention_layer*2*128插res后两层且+和_feed256_1-16-128"

        # project_name = "resnet5block-1d+attention_layer*5*128_feed512_1-16-128"
        # project_name = "resnet5block-1d_1-16-256+attention_layer*2*256_feed256"
        # project_name = "resnet7block-1d_1-16-1024+attention_layer*2_feed256、256_res4,6_no_add"
        # "resnet7block-1d_1-16-1024+attention_layer*3_feed256_res4,5,6_no_add"
        # project_name = f"resnet7block-1d_1-16-1024_attention_layer*2_feed256_46_no_add_7class-{configs.pre_train}_pretrain"
        project_name = f"{configs.dataset_type_train}_to{configs.dataset_type_test}"+model_name+f"{configs.n_way}way_{configs.k_shot}shot_{configs.pre_train}_pretrain"
        # project_name = "resnet7block-1d_1-16-1024_fft_abs"
        # project_name = "resnet-2n-8block-1-8-1024+attention_layer*1*_feed512"
        # project_name = "魔改中resnet1d+transformer_无lvbo_无flod_不同学习率_AdamW1e-6_positional_encoding"
        # lr_rates = [0.0001, 0.0005]

        for lr_rate in lr_rates:
            best_acc = main(lr_rate, project_name, configs)
            best_Score_record.append(best_acc)

        print("-" * 20 + project_name + "-" * 20)

        # for lr_rate, best_score in zip(lr_rates, best_Score_record):
        #     print(f"lr_rate: {lr_rate}  best_acc: {best_score[0]:.4f} best_recall: {best_score[1]:.4f}  best_precision: {best_score[2]:.4f}  best_f1: {best_score[3]:.4f}  best_roc_auc: {best_score[4]:.4f}")

        # 创建文件夹
        output_dir = f"resnet1d-senet-7block_few_shot"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # 打开文件以写入
        # output_name = f"resnet-senet-7block.txt  resnet-2n-8block"
        output_name = f"{configs.dataset_type_train}_to{configs.dataset_type_test}"+model_name+f"{configs.n_way}way_{configs.k_shot}shot.txt"
        output_file = os.path.join(output_dir, output_name)
        with open(output_file, "a") as file:
            file.write("-" * 20 + project_name + "-" * 20 + "\n")

            for lr_rate, best_score in zip(lr_rates, best_Score_record):
                # file.write(f"lr_rate: {lr_rate}  best_acc: {best_score[0]:.4f} best_recall: {best_score[1]:.4f}  best_precision: {best_score[2]:.4f}  best_f1: {best_score[3]:.4f}  best_roc_auc: {best_score[4]:.4f}\n")
                file.write(f"lr_rate: {lr_rate}  best_acc: {best_score:.4f}\n")
            file.write(project_name + "\n")