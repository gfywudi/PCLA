import pickle
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from single_lead_config import get_train_config
from load_data_single_lead import load_data

from imblearn.over_sampling import ADASYN,SMOTE
import subprocess
import signal
# from single_lead_data_loader import augment_data
# from Network import LogisticRegressionModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

import os
import sys
import time

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from cao_paper.mamba_CNN import Model_routine
from RSTAnet import ResNet1D,ResNet1DBlock



from cao_paper.prototypicalNetwork import PrototypicalNetwork

from sklearn import preprocessing

import torch.backends.cudnn as cudnn
# import BalancedDataParallel
# from tqdm import tqdm
# from util import adjust_learning_rate, accuracy, AverageMeter
# import wandb
from torch.autograd import Variable

from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, Precision, Recall, AUROC, F1Score
# 除去 precision的报错
import warnings
warnings.filterwarnings("ignore")
import time
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
from siamese_network import SiameseNetwork


def dimcon(x):
    t = []
    x_pin = torch.tensor(t)
    for i in range(1):
        x_er = x[:, :, i]
        x_era = preprocessing.scale(x_er)#z分数归一化
        x_era = torch.Tensor(x_era)
        x_era = Variable(x_era)
        x_era = x_era.unsqueeze(1)  # 将x的维度进行扩展
        x_pin = torch.cat((x_pin, x_era), dim=1, out=None)
    return x_pin


class MetaLearningDataset(Dataset):
    def __init__(self, data,test_data, n_ways, n_shots, n_queries, num_tasks, train_or_test ,transform=None):
        """
        初始化 MetaLearningDataset

        参数：
        - data: 数据字典，键为类别，值为该类别下的样本列表
        - n_ways: 每个任务中的类别数
        - n_shots: 每个类别用于支持集的样本数
        - n_queries: 每个类别用于查询集的样本数
        - transform: 数据转换
        """
        self.data = data
        self.n_ways = n_ways
        self.n_shots = n_shots
        self.n_queries = n_queries
        self.classes = list(data.keys())
        self.transform = transform
        self.num_tasks = num_tasks
        self.tasks = []
        self.train_or_test = train_or_test
        self.test_data = test_data

        # 生成所有任务的数据
        for _ in range(num_tasks):
            support_set, support_labels, query_set, query_labels = self._generate_task()
            self.tasks.append((support_set, support_labels, query_set, query_labels))
            # print(self.tasks)
        print("tasks:", len(self.tasks))

    def __len__(self):
        """
        返回数据集的长度
        """
        # return len(self.data)
        return self.num_tasks

    def __getitem__(self, idx):
        """
        获取一个任务
        """
        support_set, support_labels, query_set, query_labels = self._generate_task()
        return support_set, support_labels, query_set, query_labels

    def _generate_task(self):
        """
        生成一个任务
        """

        if configs.dataset_type_train == "chapman":
            selected_classes = ['AFIB', 'SR', 'ST', 'SVT', 'SB', 'SI', 'AF', 'AT', 'AVNRT']#来保证cm中的顺序没问题
        elif configs.dataset_type_train == "chapman_3class":
            selected_classes = ['AFIB', 'AT', 'AVNRT']
        else:
            selected_classes = ['SR', 'AFIB', 'STACH', 'SARRH', 'SBRAD', 'PACE', 'SVARR', 'BIGU', 'AFLT']

        # 在选定的类别中随机选择 n_ways 个类别
        # selected_classes = np.random.choice(self.classes, self.n_ways, replace=False)
        # 打乱选定的类别顺序以增加随机性
        # np.random.shuffle(selected_classes)
        # selected_classes = np.random.choice(self.classes, self.n_ways, replace=False)
        support_set = []
        support_labels = []
        query_set = []
        query_labels = []

        for class_idx, cls in enumerate(selected_classes):
            # 从该类别中随机选择 n_shots + n_queries 个样本
            samples_id = np.random.choice(len(self.data[cls]), self.n_shots + self.n_queries, replace=False)
            np.random.shuffle(samples_id)
            task_data = [(self.data[cls][i], class_idx) for i in samples_id]


            test_samples_id = np.random.choice(len(self.test_data[cls]), len(self.test_data[cls]),replace=False)
            np.random.shuffle(samples_id)
            test_task_data = [(self.test_data[cls][i], class_idx) for i in test_samples_id]

            # 分配支持集和查询集及其标签
            if self.train_or_test == "train":
                for i, (sample, label) in enumerate(task_data):
                    if self.transform:
                        sample = self.transform(sample)

                    if i < self.n_shots:
                        # 前 n_shots 个样本用于支持集
                        support_set.append(sample)
                        support_labels.append(label)
                    else:
                        # 剩下的用于查询集
                        query_set.append(sample)
                        query_labels.append(label)

            elif self.train_or_test == "test":
                for i, (sample, label) in enumerate(task_data):
                    if self.transform:
                        sample = self.transform(sample)

                    if i < self.n_shots:
                        # 前 n_shots 个样本用于支持集
                        support_set.append(sample)
                        support_labels.append(label)

                    else:
                        for j, (sample_test, label_test) in enumerate(test_task_data):
                            query_set.append(sample_test)
                            query_labels.append(label_test)


        # 转换列表为张量
        support_set = torch.stack(support_set)
        support_labels = torch.tensor(support_labels)
        query_set = torch.stack(query_set)
        query_labels = torch.tensor(query_labels)
        return support_set, support_labels, query_set, query_labels


def smote(X,y):
    # 初始化一个列表以存储每个特征维度的结果
    X_resampled_list = []
    y_resampled_list = []
    #(131,5000,12)
    # 遍历每个特征维度（共12个）
    if np.isnan(X).any():
        print("数据中存在 NaN 值，进行处理...")
        # 使用均值填充 NaN 值
        # 注意：为了避免在计算均值时引入 NaN，先将 NaN 替换为 0
        X = np.nan_to_num(X, nan=0)  # 将 NaN 替换为 0

    # print("X.shape[1]", X.shape[1])
    for i in range(X.shape[2]):
        # 获取当前特征的所有样本
        current_feature_data = X[:, :, i]  # 形状为 (28391, 5000)
        print("current_feature_data:",current_feature_data.shape)
        # 使用 SMOTE 进行过采样
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        current_feature_resampled, current_y_resampled = smote.fit_resample(current_feature_data, y)

        # 将当前特征的结果存入列表
        X_resampled_list.append(current_feature_resampled)
        y_resampled_list.append(current_y_resampled)

    # 将每个特征维度的结果堆叠回一个三维数组
    X_resampled = np.stack(X_resampled_list, axis=1)  # 形状为 (新样本数, 12, 5000)
    y_resampled =current_y_resampled  # 连接所有标签

    return X_resampled,y_resampled

def few_data_select(labels):
    # 获取所有类别的唯一值
    train_indices = []
    val_indices = []
    unique_elements = np.unique(labels)
    # 对每个类别进行操作
    for index,class_label in enumerate(unique_elements):  # 7个类别
        # 获取当前类别的所有样本索引
        class_indices = np.where(labels == class_label)
        class_indices = class_indices[0].tolist()
        # 随机打乱当前类别的样本索引
        np.random.shuffle(class_indices)

        if len(class_indices)>100:
        # 选择前5个作为训练集，剩下的作为验证集
            a_num=34
            train_indices.extend(class_indices[a_num:a_num+6])
            val_indices.extend(class_indices[:a_num]+class_indices[a_num+6:])
        else:
            train_indices.extend(class_indices[:6])
            val_indices.extend(class_indices[6:])

    # 保存当前训练集和验证集的索引

    return train_indices,val_indices

def score_z_norm(data):
    newData = np.zeros((data.shape))
    for i in range(1):
        mean = np.mean(data[:, :, i:i+1], axis=(0, 1), keepdims=True)
        std = np.std(data[:, :, i:i+1], axis=(0, 1), keepdims=True)
        std = np.where(std == 0, 1, std)
        newData[:, :, i] = (data[:, :, i] - mean) / std
    return newData

def meta_dataset(train_data,single_value_labels):
    if configs.dataset_type_train == "chapman":
        data_dict = {}
        label_list = ['AFIB', 'SR', 'ST', 'SVT', 'SB','SI', 'AF','AT', 'AVNRT']

    elif configs.dataset_type_train == "chapman_3class":
        data_dict = {}
        label_list = ['AFIB', 'AT', 'AVNRT']

    else:
        data_dict = {}
        label_list =['SR','AFIB','STACH','SARRH','SBRAD','PACE', 'SVARR','BIGU','AFLT']


        # z分数归一化
    cleaned_data_Z_score = score_z_norm(train_data)

    for j in range(len(label_list)):
        data_dict[label_list[j]] = []
    for i in range(len(single_value_labels)):
        data_dict[label_list[single_value_labels[i]]].append(torch.tensor(cleaned_data_Z_score[i:i + 1, :, :]))

    def print_class_distribution(y, dataset_name):
        unique, counts = np.unique(y, return_counts=True)
        print(f"{dataset_name} 的类别分布数量:")
        for label, count in zip(unique, counts):
            print(f"类别 {label}: {count} 个样本")

    print_class_distribution(single_value_labels, f"训练集（样本的类别）")

        # data = {k: v for k, v in data_dict.items() if v}
        # train_data = {k: v for k, v in data_dict.items() if k in A}

        # 创建 MetaLearningDataset 实例




    return data_dict






def main(lr_rate, project_name,configs):
    # 初始化
    np.random.seed(42)
    configs.fold_lr = lr_rate
    # configs.epoch = num_epochs

    best_model = None
    best_val_score = 0.0

    data_size = int(5000 / configs.data_dim)

    # 用于存储每个折叠的训练数据的损失和指标

    # iterations = configs.lr_decay_epochs.split(',')
    # configs.lr_decay_epochs = list([])
    # for it in iterations:
    #     configs.lr_decay_epochs.append(int(it))


    np.set_printoptions(linewidth=400, precision=4)

    '''set device'''


    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    '''load data'''
    data_iter = load_data(configs)
    print('=' * 20, 'load data over', '=' * 20)

    # 将数据分割成训练集和剩余集（包括验证集和测试集）
    # X_train_o, X_test, y_train_o, y_test = train_test_split(data_iter[0], data_iter[1], test_size=0.2, random_state=42)
    X_train= data_iter[0]
    y_train= data_iter[1]
    X_train_few = data_iter[2]
    y_train_few = data_iter[3]#这些y值都不是onehot的形式
    if np.isnan(X_train).any():
        print("数据中存在 NaN 值，进行处理...")
        # 使用均值填充 NaN 值
        # 注意：为了避免在计算均值时引入 NaN，先将 NaN 替换为 0
        X_train = np.nan_to_num(X_train, nan=0)  # 将 NaN 替换为 0
    # lead = [1]
    # X_train = X_train_few[:,:,lead]
    # print(np.array(X_train_o).shape)
    # X_train, y_train = smote(X_train, y_train)
    # adasyn = ADASYN(random_state=42)
    # X_train_o = np.squeeze(X_train_o)
    # X_train, y_train = adasyn.fit_resample(X_train_o, y_train_o)
    # X_train = np.expand_dims(X_train, axis=2)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print('=' * 50 + 'Start Training' + '=' * 50)
    Fold_result = {}
    results = {}

    for fold, (train_index, val_index) in enumerate(kf.split(y_train)):
        best_val_score = 0
        best_val_f1 = 0
        best_val_roc = 0
        best_val_rec = 0
        best_val_pre = 0
        best_val_spec = 0
        X_train_many, X_val_many = X_train[train_index], X_train[val_index]
        y_train_many, y_val_many = y_train[train_index],y_train[val_index]



        train_index_few, val_index_few = few_data_select(y_train_few)
        X_train_few_fold, X_val_few_fold = X_train_few[train_index_few], X_train_few[val_index_few]
        y_train_few_fold, y_val_few_fold = y_train_few[train_index_few], y_train_few[val_index_few]
        # np.save(str(configs.dataset_type)+str(other_number)+'_label_y_test_fold_'+str(fold)+'.npy', y_val_few_fold)


        # y_train_few_fold[:] = other_number
        # y_val_few_fold[:] = other_number
        X_train_fold = np.concatenate((X_train_many, X_train_few_fold), axis=0)
        y_train_fold = np.concatenate((y_train_many, y_train_few_fold), axis=0)
        X_val_fold = np.concatenate((X_val_many, X_val_few_fold), axis=0)
        y_val_fold = np.concatenate((y_val_many, y_val_few_fold), axis=0)
        X_test_fold = X_val_few_fold
        y_test_fold = y_val_few_fold
        # np.save(str(configs.dataset_type)+str(other_number)+'_label_X_test_fold_'+str(fold)+'.npy', X_test_fold)
        print(np.array(X_val_fold).shape)
        print(np.array(y_val_fold).shape)

        # X_train_fold,y_train_fold = smote(X_train_fold, y_train_fold)
        # adasyn = SMOTE(random_state=42)
        # X_train_fold = np.squeeze(X_train_fold)
        # X_train_fold, y_train_fold = adasyn.fit_resample(X_train_fold, y_train_fold)
        # X_train_fold = np.expand_dims(X_train_fold, axis=2)
        # print(np.array(y_train_fold).shape)

        # model = Model_routine(num_classes=7)


        encoder = ResNet1D(ResNet1DBlock, [2, 2, 2, 2, 2, 2, 2], num_classes=configs.n_way).to("cuda")
        if configs.dataset_type_train =="PTB":
            pretrain_model_name = "/homeb/guofengyi/result/cao_paper/model_II_all_class/pretrain_Resmamba_no_special_treatment_7class__PTB_to_PTB_fold"+str(fold)+"_lr0.005.pth"
        else:
            pretrain_model_name = "/homeb/guofengyi/result/cao_paper/model_II_all_class/pretrain_Resmamba_no_special_treatment_7class__chapman_to_chapman_fold"+str(fold)+"_lr0.005.pth"

        if os.path.exists(pretrain_model_name):
            print("pre_train load pretrain_model_name...")
            # 加载预训练模型的状态字典
            pretrained_state_dict = torch.load(pretrain_model_name, map_location=torch.device('cuda'))
            # 获取当前模型的状态字典
            model_state_dict = encoder.state_dict()
            if hasattr(pretrained_state_dict, 'state_dict'):
                pretrained_state_dict = pretrained_state_dict.state_dict()
            print("Keys in pretrained_state_dict:", pretrained_state_dict.keys())
            print("Keys in model_state_dict:", model_state_dict.keys())

            # 从预训练状态字典中删除最后一层的权重和偏置
            pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if
                                     k in model_state_dict and "fc" not in k}
            # 更新当前模型的状态字典
            model_state_dict.update(pretrained_state_dict)
            # 加载更新后的状态字典到模型中
            encoder.load_state_dict(model_state_dict)
        pretrain_model = torch.load(pretrain_model_name)
        model = SiameseNetwork(encoder=encoder)
        # model = ResNet1D_transformer(ResNet1DBlock_transformer, [2, 2, 2, 2, 2, 2, 2], num_classes=7).to("cuda")
        # model = Model(7)
        # model = ecgTransForm()
        # model = MSDNN(7)
        # model = EffNet(output_neurons=7)
        # model = ECGModel(7)
        # model = ECGnet(num_classes=7)
        # model = AGSX(n_classes=7)



        final_optimizer = torch.optim.AdamW(model.parameters(), lr=configs.fold_lr, weight_decay=1e-6)
        # optimizer = torch.optim.Adam(model.parameters(), lr=configs.fold_lr)
        loss_fc = nn.CrossEntropyLoss()
        criterion = nn.CrossEntropyLoss()
        # loss_fc = nn.BCELoss()

        if torch.cuda.is_available():
            model = model.cuda()
            criterion = loss_fc.cuda()
            cudnn.benchmark = True

        if configs.n_way == 3:
            y_train_fold = np.maximum(y_train_fold-6, 0)
            y_val_fold = np.maximum(y_val_fold-6, 0)
        else:
            pass
        train_dataset_ = meta_dataset(X_train_fold,y_train_fold)
        val_dataset_ = meta_dataset(X_val_fold, y_val_fold)

        train_dataset_1 = MetaLearningDataset(train_dataset_,val_dataset_, n_ways=configs.n_way, n_shots=configs.k_shot, n_queries=1,
                                        num_tasks=num_tasks,train_or_test="train", transform=None)
        val_dataset_1 = MetaLearningDataset(train_dataset_, val_dataset_, n_ways=configs.n_way, n_shots=configs.k_shot, n_queries=1,
                                            num_tasks=1, train_or_test="test", transform=None)
        train_dataset = DataLoader(train_dataset_1, batch_size=configs.train_batch_size, shuffle=True, num_workers=0)
        val_dataset = DataLoader(val_dataset_1, batch_size=configs.test_batch_size, shuffle=True, num_workers=0)

        # 创建数据加载器
    #     train_dataset = TensorDataset(dimcon(X_train_fold),
    #                                   torch.tensor(y_train_fold, dtype=torch.long))
    #     train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True,num_workers=4)
    #     print("完成归一化，将要进行训练")
    #     val_dataset = TensorDataset(dimcon(X_val_fold),
    #                                 torch.tensor(y_val_fold, dtype=torch.long))
    #     val_loader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False,num_workers=4)
    #     print("完成归一化，将要进行训练")

        fold_train_loss = []


        # 在训练集上进行模型训练
        for epoch in range(configs.epoch):
            # scheduler.step()
            model.train()
            correct_all = 0
            total = 0
            loop = tqdm(enumerate(train_dataset), total=len(train_dataset))
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
                query_labels_one_hot = F.one_hot(query_labels, num_classes=9).float()
                support_labels_onehot = F.one_hot(support_labels, num_classes=9).float()
                similarities = model(support_set, query_set, support_labels_onehot)

                # 计算损失
                loss = criterion(similarities, query_labels_one_hot)
                # print("similarities",similarities.shape)
                loss.backward()
                final_optimizer.step()

                # print("outputs", outputs.size(),outputs)
                # 应用 softmax 函数将输出转换为概率分布
                # probabilities = F.softmax(outputs, dim=1)
                # 计算准确率
                # final_batch_metrics = final_train_metrics.cuda().forward(dists, query_labels)
                # print("similarities",similarities.shape)
                _, preds = torch.max(similarities, dim=1)
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

                loop.set_description(f"train Epoch: [{epoch}][{idx}/{len(train_dataset)}]\t")
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
                fold_valid_output = []
                pretrain_label_list = []
                pretrain_predict_list = []
                pretrain_fold_valid_output = []
                avg_test_accuracy = 0.0
                avg_test_loss = 0.0

                test_correct_all = 0
                test_total = 0
                model.eval()
                with torch.no_grad():
                    # loop = tqdm(enumerate(final_test_loader), total=len(final_test_loader))
                    # for idx, batch in loop:valid
                    for idx, batch in enumerate(val_dataset):
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
                        # print(query_labels)
                        pretrain_output = pretrain_model(query_set)
                        _, pretrain_preds = torch.max(pretrain_output, dim=1)
                        pretrain_predict_list.append(pretrain_preds.squeeze().int().cpu().detach().numpy().tolist())                        # 前向传播和计算损失
                        softmax = nn.Softmax(dim=1)
                        pretrain_fold_valid_output.append(softmax(pretrain_output).squeeze().cpu().detach().numpy().tolist())

                        # 前向传播
                        query_labels_one_hot = F.one_hot(query_labels, num_classes=9).float()
                        support_labels_onehot = F.one_hot(support_labels, num_classes=9).float()
                        similarities = model(support_set, query_set, support_labels_onehot)
                        # 计算损失
                        # print("similarities",similarities.shape)
                        # print("query_labels", query_labels.shape)
                        # similarities = model(support_set, support_labels, query_set)
                        # 计算损失
                        loss = criterion(similarities, query_labels_one_hot)                        # print("outputs", outputs.size(),outputs)
                        # 应用 softmax 函数将输出转换为概率分布
                        # probabilities = F.softmax(outputs, dim=1)
                        _, preds = torch.max(similarities, dim=1)
                        predict_list.append(preds.squeeze().int().cpu().detach().numpy().tolist())
                        label_list.append(query_labels.int().cpu().detach().numpy().tolist())
                        fold_valid_output.append(softmax(similarities).squeeze().cpu().detach().numpy().tolist())
                        # for i in range(len(preds.squeeze().int().cpu().detach().numpy().tolist())):
                        #     if preds.squeeze().int().cpu().detach().numpy().tolist()[i] == 0 or 1 or 2 or 3 or 4 or 5:
                        #         pretrain_output = pretrain_model(query_set)
                        #         _, pretrain_preds = torch.max(pretrain_output, dim=1)
                        #         preds.squeeze().int().cpu().detach().numpy().tolist()[i] = pretrain_preds
                        #         print(pretrain_preds[i])
                        #     else:
                        #         pass
                        test_correct = (preds == query_labels).sum().item()
                        test_correct_all += test_correct
                        test_total += query_labels.size(0)


                    test_accuracy = test_correct_all / test_total

                    print(
                        f'Epoch {epoch + 1}/{configs.epoch}, test Accuracy: {test_accuracy:.4f}, test loss: {loss.item():.4f}:')


                    # 计算平均准确率
                    def to_one_hot(labels, num_classes):
                        return np.eye(num_classes)[labels]

                    labels = [0, 1, 2,3,4,5,6,7,8]
                    merged_label_list = np.array(sum(label_list,[]))
                    merged_predict_list = sum(predict_list, [])
                    merged_valid_output= np.squeeze(np.array(fold_valid_output))
                    fold_train_label_onehot = to_one_hot(merged_label_list, num_classes=configs.n_way)
                    # merged_predict_list = posterior_probability(fold_train_label_onehot,merged_valid_output,merged_predict_list)
                    # merged_predict_list, best_threshold_7, best_threshold_8 = predict_with_optimal_thresholds(
                    #     fold_train_label_onehot, merged_valid_output)
                    # predict_indices_not_8_9 = [i for i, value in enumerate(merged_predict_list) if
                    #                            value != configs.n_way - 2 and value != configs.n_way - 1]
                    merged_predict_list = np.squeeze(np.array(pretrain_predict_list))
                    # print(best_threshold_7)
                    # print(best_threshold_8)
                    avg_val_accuracy = accuracy_score(merged_label_list, merged_predict_list)
                    avg_val_pre = precision_score(merged_label_list, merged_predict_list, average='weighted')
                    avg_val_rec = recall_score(merged_label_list, merged_predict_list, average='weighted')
                    avg_val_f1 = f1_score(merged_label_list, merged_predict_list, average='weighted')
                    avg_val_roc = calculate_auc_not_8_9(fold_train_label_onehot,   np.squeeze(np.array(pretrain_fold_valid_output)),n_classes=7)
                    # cm = confusion_matrix(merged_label_list[label_indices_not_8_9], merged_predict_list[label_indices_not_8_9])
                    cm_all = confusion_matrix(merged_label_list, merged_predict_list)
                    avg_val_spec = macro_specificity_all_class(cm_all)
                    print("cm:", cm_all)


                    def to_one_hot_8_9(labels):
                        unique_labels = sorted(set(labels))

                        # 创建一个字典将每个标签映射到一个索引
                        label_to_index = {label: index for index, label in enumerate(unique_labels)}
                        # 根据字典将每个标签转换为对应的 one-hot 向量
                        one_hot_encoded = np.array(
                            [[1 if i == label_to_index[label] else 0 for i in range(len(unique_labels))] for label in
                             labels])
                        return one_hot_encoded

                    indices_8_9 = [i for i, value in enumerate(merged_label_list) if value == configs.n_way-2 or value == configs.n_way-1]

                    counts = np.count_nonzero((merged_predict_list == configs.n_way - 2) | (merged_predict_list == configs.n_way - 1))#merged_predict_list是一个array
                    wrong_indice = counts - len(indices_8_9)
                    print(wrong_indice)

                    merged_label_list_8_9 = [merged_label_list[i] for i in indices_8_9]
                    merged_predict_list_8_9 = [merged_predict_list[i] for i in indices_8_9]

                    avg_val_accuracy_8_9 = accuracy_score(merged_label_list_8_9, merged_predict_list_8_9)
                    avg_val_pre_8_9 = macro_precision(merged_label_list, merged_predict_list)
                    avg_val_rec_8_9 = macro_recall(merged_label_list, merged_predict_list)
                    avg_val_f1_8_9 = macro_f1(merged_label_list, merged_predict_list)
                    avg_val_roc_8_9 = calculate_auc_8_9(merged_label_list, merged_valid_output, n_classes=configs.n_way)
                    cm_8_9 = confusion_matrix(merged_label_list, merged_predict_list)
                    avg_val_spec_8_9 = macro_specificity(cm_8_9)

                    # print(t)
                    # avg_val_roc = roc_auc_score(fold_train_label, fold_train_pred, average='weighted')
                    # print(t)j

                    # avg_val_accuracy = sum(val_accuracies) / len(val_accuracies)
                    # avg_train_loss = sum(fold_train_loss) / len(fold_train_loss)
                    # avg_val_f1 = sum(val_f1) / len(val_f1)
                    # avg_val_roc = sum(val_roc) / len(val_roc)
                    # avg_val_rec = sum(val_rec) / len(val_rec)
                    # avg_val_pre = sum(val_prec) / len(val_prec)

                    # print(f"valid loss (Fold {fold + 1}):",avg_train_loss)
                    # print(f"Validation Score_few (Fold {fold + 1}):", avg_val_accuracy_8_9, avg_val_f1_8_9, avg_val_roc_8_9,avg_val_rec_8_9,avg_val_pre_8_9)
                    # print(f"Validation Score_all (Fold {fold + 1}):", avg_val_accuracy, avg_val_f1, avg_val_roc,avg_val_rec,avg_val_pre)
                    print(f"Validation Score_few (Fold {fold + 1}):\t"
                          f"best val score: {avg_val_accuracy_8_9:.4f}\t"
                          f""f"val_f1:{avg_val_f1_8_9}\t"
                          f"val_roc:{avg_val_roc_8_9}\t"
                          f"val_rec:{avg_val_rec_8_9}\t"
                          f"val_prec:{avg_val_pre_8_9}\t"
                          f"val_spec:{avg_val_spec_8_9}\t")
                    print(f"Validation Score_all (Fold {fold + 1}):\t"
                          f"best val score: {avg_val_accuracy:.4f}\t"
                          f""f"val_f1:{avg_val_f1}\t"
                          f"val_roc:{avg_val_roc}\t"
                          f"val_rec:{avg_val_rec}\t"
                          f"val_prec:{avg_val_pre}\t"
                          f"val_spec:{avg_val_spec}\t")
                    # 如果当前模型的验证得分更好，更新最佳模型
                    if avg_val_accuracy_8_9 > best_val_score:
                    # if avg_val_accuracy > best_val_score:
                        best_model = model
                        # best_val_score = avg_val_accuracy
                        # best_val_f1 = avg_val_f1
                        # best_val_roc = avg_val_roc
                        # best_val_rec = avg_val_rec
                        # best_val_pre = avg_val_pre
                        # hunxiao  = cm
                        best_val_score = avg_val_accuracy_8_9
                        best_val_f1 = avg_val_f1_8_9
                        best_val_roc = avg_val_roc_8_9
                        best_val_rec = avg_val_rec_8_9
                        best_val_pre = avg_val_pre_8_9
                        best_val_spec = avg_val_spec_8_9
                        hunxiao = cm_8_9
                        best_val_score_many = avg_val_accuracy
                        best_val_f1_many = avg_val_f1
                        best_val_roc_many = avg_val_roc
                        best_val_rec_many = avg_val_rec
                        best_val_pre_many = avg_val_pre
                        best_val_spec_many = avg_val_spec
                        torch.save(best_model, "/homeb/guofengyi/result/cao_paper/model_II_all_class/"+str(project_name)+'_fold'+str(fold)+"_lr"+str(lr_rate)+'.pth')
                        print(f"change best val score: {best_val_score:.4f}\t"f"val_f1:{best_val_f1}\t"
                              f"val_roc:{best_val_roc}\t"
                              f"val_rec:{best_val_rec}\t"
                              f"val_prec:{best_val_pre}\t"
                              f"val_spec:{best_val_spec}\t")
                    else:
                        print(f"Validation Score (Fold {fold + 1}):\t"
                            f"best val acc score: {best_val_score:.4f}\t"
                              f"val_f1:{best_val_f1}\t"
                              f"val_roc:{best_val_roc}\t"
                              f"val_rec:{best_val_rec}\t"
                              f"val_prec:{best_val_pre}\t"
                              f"val_spec:{best_val_spec}\t")
        Fold_result["acc"] = best_val_score
        Fold_result["f1"] = best_val_f1
        Fold_result["roc"] = best_val_roc
        Fold_result["rec"] = best_val_rec
        Fold_result["prec"] = best_val_pre
        Fold_result["spec"] = best_val_spec
        Fold_result["hunxiao"] = np.array2string(hunxiao, separator=', ')

        Fold_result["acc_many"] = best_val_score_many
        Fold_result["f1_many"] = best_val_f1_many
        Fold_result["roc_many"] = best_val_roc_many
        Fold_result["rec_many"] = best_val_rec_many
        Fold_result["prec_many"] = best_val_pre_many
        Fold_result["spec_many"] = best_val_spec_many



        for key, value in Fold_result.items():
            if key in results:
                results[key].append(value)   # 拼接值
            else:
                results[key] = []  # 初始化值
                results[key].append(value)

        with open("/homeb/guofengyi/result/cao_paper/result_II_all_class/"+str(project_name)+".txt", 'a') as file:
            file.write('\n' + "learning_rate:" + str(lr_rate) +'\n')
            file.write('\n' + "the number:" + str(fold) + "zhe" + '\n')
            file.write("acc_many:" + str(Fold_result["acc_many"]) + '\n')
            file.write("f1_many:" + str(Fold_result["f1_many"]) + '\n')
            file.write("roc_many:" + str(Fold_result["roc_many"]) + '\n')
            file.write("rec_many:" + str(Fold_result["rec_many"]) + '\n')
            file.write("prec_many:" + str(Fold_result["prec_many"]) + '\n')
            file.write("spec_many:" + str(Fold_result["spec_many"]) + '\n')
            file.write("hunxiao:" + Fold_result["hunxiao"] + '\n')

            file.write("acc_few:" + str(Fold_result["acc"]) + '\n')
            file.write("f1_few:" + str(Fold_result["f1"]) + '\n')
            file.write("roc_few:" + str(Fold_result["roc"]) + '\n')
            file.write("rec_few:" + str(Fold_result["rec"]) + '\n')
            file.write("prec_few:" + str(Fold_result["prec"]) + '\n')
            file.write("spec_few:" + str(Fold_result["spec"]) + '\n')

 # 转换为 NumPy 数组
    avg_results_many = {
        "acc": np.mean(np.array(results[ "acc_many"])* 100),
        "f1": np.mean(np.array(results[ "f1_many"])* 100),
        "roc": np.mean(np.array(results[ "roc_many"])* 100),
        "rec": np.mean(np.array(results[ "rec_many"])* 100),
        "prec": np.mean(np.array(results[ "prec_many"])* 100),
        "spec": np.mean(np.array(results["spec_many"]) * 100),
    }
    var_results_many = {
        "acc": np.var(np.array(results[ "acc_many"])* 100),
        "f1": np.var(np.array(results[ "f1_many"])* 100),
        "roc": np.var(np.array(results[ "roc_many"])* 100),
        "rec": np.var(np.array(results[ "rec_many"])* 100),
        "prec": np.var(np.array(results[ "prec_many"])* 100),
        "spec": np.mean(np.array(results["spec_many"]) * 100),
    }
    std_results_many = {
        "acc": np.std(np.array(results[ "acc_many"])* 100),
        "f1": np.std(np.array(results[ "f1_many"])* 100),
        "roc": np.std(np.array(results[ "roc_many"])* 100),
        "rec": np.std(np.array(results[ "rec_many"])* 100),
        "prec": np.std(np.array(results[ "prec_many"])* 100),
        "spec": np.mean(np.array(results["spec_many"]) * 100),
    }

    avg_results = {
        "acc": np.mean(np.array(results["acc"]) * 100),
        "f1": np.mean(np.array(results["f1"]) * 100),
        "roc": np.mean(np.array(results["roc"]) * 100),
        "rec": np.mean(np.array(results["rec"]) * 100),
        "prec": np.mean(np.array(results["prec"]) * 100),
        "spec": np.mean(np.array(results["spec"]) * 100),
    }
    var_results = {
        "acc": np.var(np.array(results["acc"]) * 100),
        "f1": np.var(np.array(results["f1"]) * 100),
        "roc": np.var(np.array(results["roc"]) * 100),
        "rec": np.var(np.array(results["rec"]) * 100),
        "prec": np.var(np.array(results["prec"]) * 100),
        "spec": np.mean(np.array(results["spec"]) * 100),
    }
    std_results = {
        "acc": np.std(np.array(results["acc"]) * 100),
        "f1": np.std(np.array(results["f1"]) * 100),
        "roc": np.std(np.array(results["roc"]) * 100),
        "rec": np.std(np.array(results["rec"]) * 100),
        "prec": np.std(np.array(results["prec"]) * 100),
        "spec": np.mean(np.array(results["spec"]) * 100),
    }



    # 将平均值、方差和标准差写入文件
    with open("/homeb/guofengyi/result/cao_paper/result_II_all_class/" + str(project_name) + ".txt", 'a') as file:
        file.write('\n' + "learning_rate:" + str(lr_rate) + '\n')
        file.write('\n' + "average_many" + '\n')
        file.write(
            "acc: mean = " + str(avg_results_many["acc_many"]) + ", variance = " + str(var_results_many["acc_many"]) + ", std = " + str(
                std_results_many["acc_many"]) + '\n')
        file.write("f1: mean = " + str(avg_results_many["f1"]) + ", variance = " + str(var_results_many["f1"]) + ", std = " + str(
            std_results_many["f1"]) + '\n')
        file.write(
            "roc: mean = " + str(avg_results_many["roc"]) + ", variance = " + str(var_results_many["roc"]) + ", std = " + str(
                std_results_many["roc"]) + '\n')
        file.write(
            "rec: mean = " + str(avg_results_many["rec"]) + ", variance = " + str(var_results_many["rec"]) + ", std = " + str(
                std_results_many["rec"]) + '\n')
        file.write(
            "prec: mean = " + str(avg_results_many["prec"]) + ", variance = " + str(var_results_many["prec"]) + ", std = " + str(
                std_results_many["prec"]) + '\n')
        file.write(
            "spec: mean = " + str(avg_results_many["spec"]) + ", variance = " + str(var_results_many["spec"]) + ", std = " + str(
                std_results_many["spec"]) + '\n')

        file.write('\n' + "average_few" + '\n')
        file.write(
            "acc: mean = " + str(avg_results["acc"]) + ", variance = " + str(var_results["acc"]) + ", std = " + str(
                std_results["acc"]) + '\n')
        file.write("f1: mean = " + str(avg_results["f1"]) + ", variance = " + str(var_results["f1"]) + ", std = " + str(
            std_results["f1"]) + '\n')
        file.write(
            "roc: mean = " + str(avg_results["roc"]) + ", variance = " + str(var_results["roc"]) + ", std = " + str(
                std_results["roc"]) + '\n')
        file.write(
            "rec: mean = " + str(avg_results["rec"]) + ", variance = " + str(var_results["rec"]) + ", std = " + str(
                std_results["rec"]) + '\n')
        file.write(
            "prec: mean = " + str(avg_results["prec"]) + ", variance = " + str(var_results["prec"]) + ", std = " + str(
                std_results["prec"]) + '\n')
        file.write(
            "spec: mean = " + str(avg_results["spec"]) + ", variance = " + str(var_results["spec"]) + ", std = " + str(
                std_results["spec"]) + '\n')
    print("Average Results:")
    print(avg_results)
    print("Variance Results:")
    print(var_results)
    print("Standard Deviation Results:")
    print(std_results)

    return best_val_score


# 计算每个类别的特异性
def macro_specificity_all_class(cm):
    specificities = []
    for i in range(configs.n_way-2):
        # 获取当前类别的 TN 和 FP
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]  # 真阴性
        fp = np.sum(cm[:, i]) - cm[i, i]  # 假阳性
        specificity = tn / (tn + fp) if tn + fp != 0 else 0  # 避免除零错误
        specificities.append(specificity)
    return np.mean(specificities)

def macro_specificity(cm):
    specificities = []
    for i in range(configs.n_way):
        # 获取当前类别的 TN 和 FP
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]  # 真阴性
        fp = np.sum(cm[:, i]) - cm[i, i]  # 假阳性
        specificity = tn / (tn + fp) if tn + fp != 0 else 0  # 避免除零错误
        specificities.append(specificity)
    return np.mean(specificities[-num_rare_class:])

# 计算宏特异性
# def macro_specificity(conf_matrix):
#     specificity_per_class = specificity(conf_matrix)
#     return np.mean(specificity_per_class[-2:])

def macro_recall(y_true, y_pred):
    # 获取所有类别
    classes = np.unique(y_true)

    # 初始化 True Positives 和 False Negatives
    recall_per_class = []

    for cls in classes:
        # 获取当前类别的 TP 和 FN
        TP = np.sum((y_true == cls) & (y_pred == cls))  # 真实类别为cls且预测为cls
        FN = np.sum((y_true == cls) & (y_pred != cls))  # 真实类别为cls但预测为其他类别

        # 防止除以0的错误，添加一个小的epsilon值
        if TP + FN > 0:
            recall = TP / (TP + FN)
        else:
            recall = 0  # 当TP + FN = 0时，召回率为0

        recall_per_class.append(recall)

    # 计算宏观召回率（所有类别召回率的平均）
    macro_recall = np.mean(recall_per_class[-num_rare_class:])
    return macro_recall


def macro_precision(y_true, y_pred):
    # 获取所有类别
    classes = np.unique(y_true)

    # 初始化 True Positives 和 False Positives
    precision_per_class = []

    for cls in classes:
        # 获取当前类别的 TP 和 FP
        TP = np.sum((y_true == cls) & (y_pred == cls))  # 真实类别为cls且预测为cls
        FP = np.sum((y_true != cls) & (y_pred == cls))  # 真实类别不是cls但预测为cls

        # 防止除以0的错误，添加一个小的epsilon值
        if TP + FP > 0:
            precision = TP / (TP + FP)
        else:
            precision = 0  # 当TP + FP = 0时，精度为0

        precision_per_class.append(precision)

    # 计算宏观精度（所有类别精度的平均）
    macro_precision = np.mean(precision_per_class[-num_rare_class:])
    return macro_precision
def calculate_auc_not_8_9(y_true, y_pred, n_classes):
    """
    计算 Macro-AUC

    参数:
    - y_true: 真实标签 (形状: [num_samples])，每个样本的类别标签 (整数值 0 ~ num_classes-1)。
    - y_pred: 预测概率 (形状: [num_samples, num_classes])，每个样本属于每个类别的概率。
    - num_classes: 类别总数。

    返回:
    - macro_auc: Macro-AUC 的值。
    """
    y_true = np.array(y_true)
    aucs = []
    for c in range(n_classes):
        # 将类别 c 的标签转为二元标签
        # binary_true = (y_true == c).astype(int)
        # 计算类别 c 的 AUC
        try:
            auc = roc_auc_score(y_true[:,c], y_pred[:, c])
            aucs.append(auc)
        except ValueError:
            # 如果某个类别在 y_true 中没有正样本或负样本，跳过
            print(f"Warning: Class {c} has no positive or negative samples, skipping...")
            continue

    # 计算 Macro-AUC
    macro_auc = np.mean(aucs[:num_rare_class])
    return macro_auc


# 计算单个类别的 ROC 曲线
def calculate_auc_8_9(y_true, y_pred, n_classes):
    """
    计算 Macro-AUC

    参数:
    - y_true: 真实标签 (形状: [num_samples])，每个样本的类别标签 (整数值 0 ~ num_classes-1)。
    - y_pred: 预测概率 (形状: [num_samples, num_classes])，每个样本属于每个类别的概率。
    - num_classes: 类别总数。

    返回:
    - macro_auc: Macro-AUC 的值。
    """
    y_true = np.array(y_true)
    aucs = []
    for c in range(n_classes):
        # 将类别 c 的标签转为二元标签
        binary_true = (y_true == c).astype(int)
        # 计算类别 c 的 AUC
        try:
            auc = roc_auc_score(binary_true, y_pred[:, c])
            aucs.append(auc)
        except ValueError:
            # 如果某个类别在 y_true 中没有正样本或负样本，跳过
            print(f"Warning: Class {c} has no positive or negative samples, skipping...")
            continue

    # 计算 Macro-AUC
    macro_auc = np.mean(aucs[-num_rare_class:])
    return macro_auc


def macro_f1(y_true, y_pred):
    # 获取所有类别
    classes = np.unique(y_true)

    # 初始化每个类别的F1分数
    f1_per_class = []

    for cls in classes:
        # 获取当前类别的 TP, FP, FN
        TP = np.sum((y_true == cls) & (y_pred == cls))  # 真实类别为cls且预测为cls
        FP = np.sum((y_true != cls) & (y_pred == cls))  # 真实类别不是cls但预测为cls
        FN = np.sum((y_true == cls) & (y_pred != cls))  # 真实类别为cls但预测为其他类别

        # 计算精确率和召回率
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        # 计算F1分数
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0  # 如果precision和recall都为0，则F1为0

        f1_per_class.append(f1)

    # 计算Macro F1（所有类别F1的平均）
    macro_f1 = np.mean(f1_per_class[-2:])
    return macro_f1




def posterior_probability(label_list_onehot,pred_array,label_list):
    from sklearn import metrics
    best_list = []
    for i,suoyin in enumerate([7,8]):
        y_pred = label_list
        best_f1 = 0
        _, _, thresholds = metrics.roc_curve(label_list_onehot[:, suoyin], pred_array[:, suoyin])
        for threshold in thresholds:
            for j in range(len(label_list)):
                if pred_array[j,suoyin] >= threshold:
                    label_list[j] = suoyin
                    y_pred[j] = 1
                else:
                    pass
            f1 = f1_score(label_list_onehot[:, suoyin], y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold_ = threshold
                best_list = label_list
                print(best_threshold_)
    return best_list


def find_optimal_thresholds(label_list_onehot, pred_array):
    """
    自动寻找第七种和第八种标签的最优阈值，并输出最终预测结果。

    参数：
    - label_list_onehot: (numpy array) 真实标签的 one-hot 编码，形状为 (num_samples, 9)
    - pred_array: (numpy array) 模型的预测概率，形状为 (num_samples, 9)

    返回：
    - best_threshold_7: 第七种标签的最佳阈值
    - best_threshold_8: 第八种标签的最佳阈值
    """
    best_threshold_7, best_threshold_8 = 0, 0
    best_f1_7, best_f1_8,best_recall_8, best_recall_7= 0, 0, 0, 0
    from sklearn.metrics import roc_curve
    # 自动寻找第七种标签的最优阈值

    fpr, tpr, thresholds_7 = roc_curve(label_list_onehot[:, 7], pred_array[:, 7])
    for threshold in thresholds_7:
        y_pred_7 = (pred_array[:, 7] >= threshold).astype(int)
        precisions_7 = calculate_specificity(label_list_onehot[:, 7], y_pred_7)
        recalls_7 = recall_score(label_list_onehot[:, 7], y_pred_7)
        if precisions_7 >= a and recalls_7 > best_recall_7:
            best_recall_7 = recalls_7
            best_threshold_7 = threshold
        else:
            pass
        # f1_7 = f1_score(label_list_onehot[:, 7], y_pred_7)
        # if f1_7 > best_f1_7:
        #     best_f1_7 = f1_7
        #     best_threshold_7 = threshold

    # 自动寻找第八种标签的最优阈值
    fpr, tpr, thresholds_8 = roc_curve(label_list_onehot[:, 8], pred_array[:, 8])
    for threshold in thresholds_8:
        y_pred_8 = (pred_array[:, 8] >= threshold).astype(int)
        # f1_8 = f1_score(label_list_onehot[:, 8], y_pred_8)
        # if f1_8 > best_f1_8:
        #     best_f1_8 = f1_8
        #     best_threshold_8 = threshold
        precisions_8 = calculate_specificity(label_list_onehot[:, 8], y_pred_8)
        recalls_8 = recall_score(label_list_onehot[:, 8], y_pred_8)
        if precisions_8 >= a and recalls_8 > best_recall_8:
            best_recall_8 = recalls_8
            best_threshold_8 = threshold
    return best_threshold_7, best_threshold_8


def find_optimal_threshold_no_false_positives(label_list_onehot, pred_array):
    """
    寻找满足无误分类约束条件的第七类和第八类的最优阈值。

    参数：
    - label_list_onehot: (numpy array) 真实标签的 one-hot 编码，形状为 (num_samples, 9)
    - pred_array: (numpy array) 模型的预测概率，形状为 (num_samples, 9)

    返回：
    - best_threshold_7: 第七类的最佳阈值
    - best_threshold_8: 第八类的最佳阈值
    """

    def find_best_threshold_for_label(label_idx):
        """
        找到某个标签的最佳阈值，在无误分类的前提下最大化召回率。
        """
        true_labels = label_list_onehot[:, label_idx]
        pred_scores = pred_array[:, label_idx]

        # 使用 precision_recall_curve 计算所有可能的阈值
        precisions, recalls, thresholds = precision_recall_curve(true_labels, pred_scores)

        best_threshold = 0
        best_recall = 0
        for threshold in thresholds:
            # 根据阈值生成预测
            y_pred = (pred_scores >= threshold).astype(int)

            # 确保没有负类被误分类为正类
            false_positives = np.sum((y_pred == 1) & (true_labels == 0))
            if false_positives == 0:  # 精度为 1.0
                recall = np.sum((y_pred == 1) & (true_labels == 1)) / np.sum(true_labels == 1)
                if recall > best_recall:
                    best_recall = recall
                    best_threshold = threshold

        return best_threshold

    # 寻找第七类的最佳阈值
    best_threshold_7 = find_best_threshold_for_label(7)
    # 寻找第八类的最佳阈值
    best_threshold_8 = find_best_threshold_for_label(8)

    return best_threshold_7, best_threshold_8


def predict_with_optimal_thresholds(label_list_onehot, pred_array):
    """
    根据自动找到的阈值进行预测。

    参数：
    - label_list_onehot: (numpy array) 真实标签的 one-hot 编码，形状为 (num_samples, 9)
    - pred_array: (numpy array) 模型的预测概率，形状为 (num_samples, 9)

    返回：
    - final_predictions: (numpy array) 最终预测结果的类别索引，形状为 (num_samples,)
    """
    # 获取最佳阈值
    best_threshold_7, best_threshold_8 = find_optimal_thresholds(label_list_onehot, pred_array)
    # best_threshold_7, best_threshold_8 = find_optimal_threshold_no_false_positives(label_list_onehot, pred_array)
    # 初始化预测结果为概率最大的类别
    final_predictions = np.argmax(pred_array, axis=1)
    # cm = confusion_matrix(np.argmax(label_list_onehot, axis=1), final_predictions)
    # print("cm2",cm)
    # 应用第七种和第八种标签的规则
    for i in range(len(pred_array)):
        if pred_array[i, 7] > best_threshold_7:
            final_predictions[i] = 7
        elif pred_array[i, 8] > best_threshold_8:
            final_predictions[i] = 8
    cm = confusion_matrix(np.argmax(label_list_onehot, axis=1), final_predictions)
    # print("cm3",cm)
    return final_predictions, best_threshold_7, best_threshold_8


def calculate_specificity(y_true, y_pred):
    """
    计算模型的 Specificity（特异性）。

    参数:
    y_true: list 或 array，实际的标签（真实值）
    y_pred: list 或 array，模型预测的标签

    返回:
    specificity: 特异性值
    """
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 计算 Specificity
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0  # 防止除以0

    return specificity



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]='3'

    configs = get_train_config()
    best_Score_record = []
    # lr_rates = [0.001,0.005,0.01,0.05]
    lr_rates = [0.0001]
    # lr_rates = [0.01, 0.05, 0.1, 0.5]
    configs.dataset_type = 'PTB'#'PTB,chapman'
    configs.dataset_type_train = "PTB"#'PTB,chapman,chapman_3class'
    configs.batch_size = 512
    configs.epoch = 200
    if configs.dataset_type_train =="chapman_3class":
        configs.n_way = 3  # 类别
        a = 0.97
    elif configs.dataset_type_train == "chapman":
        configs.n_way = 9
        a = 0.97
    elif configs.dataset_type_train == "PTB":
        configs.n_way = 9
        a = 0.995
    # configs.n_way = 3#类别
    configs.k_shot = 5
    num_tasks = 200
    num_rare_class = 2
    configs.cuda = True#是否使用GPU，是用
    project_name = "siamese_Resmamba_combine_label_"+"_"+configs.dataset_type+"_to_"+configs.dataset_type+"_"+str(num_tasks)
    # lr_rates = [0.0001, 0.0005]2True，否用False
    # configs.device = "3"
    # if configs.dataset_type_train =="PTB":
    #     pretrain_model_road = "/homeb/guofengyi/result/cao_paper/model_II_all_class/pretrain_Resmamba_no_special_treatment_6class__PTB_to_PTB_fold0_lr0.005.pth"
    # else:
    #     pretrain_model_road = "/homeb/guofengyi/result/cao_paper/model_II_all_class/pretrain_Resmamba_no_special_treatment_7class__chapman_to_chapman_fold2_lr0.005.pth"

    with open("/homeb/guofengyi/result/cao_paper/result_II_all_class/" + str(project_name) + ".txt", 'w') as file:
        pass#w清除之前的，a不清除之前的
    for lr_rate in lr_rates:
        print(project_name)
        best_Score = main(lr_rate, project_name,configs)
        best_Score_record.append(best_Score)

    for lr_rate, best_score in zip(lr_rates, best_Score_record):
        print(f"lr_rate: {lr_rate}  best_Score: {best_score}")

    print("无滤波")




