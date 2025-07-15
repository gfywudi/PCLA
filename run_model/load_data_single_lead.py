# import pandas as pd
import os
import numpy as np
import torch

from sklearn.preprocessing import normalize
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from scipy import signal
from sklearn import preprocessing
from torch.autograd import Variable

# from imblearn.over_sampling import ADASYN

import numpy as np


def filter_samples_by_label(data, labels, target_label, num_samples, seed=42):
    # 筛选出标签为target_label的样本的索引
    target_indices = np.where(labels == target_label)[0]
    # 筛选出其他类别的样本的索引
    other_indices = np.where(labels != target_label)[0]
    # 设置随机种子
    np.random.seed(seed)
    # 随机选择num_samples个索引
    selected_indices = np.random.choice(target_indices, num_samples, replace=False)
    # 合并选定的索引和其他类别的索引
    combined_indices = np.concatenate((selected_indices, other_indices))
    # 根据合并后的索引筛选出样本和标签
    filtered_data = data[combined_indices]
    filtered_labels = labels[combined_indices]
    return filtered_data, filtered_labels



def load_data(config):
    # Data_root = '/home/caozy/project/data/ECG_data/12_lead_ECG/ECGDataDenoised'
    # Data_path = '/home/caozy/project/data/ECG_data/12_lead_ECG/Diagnostics.xlsx'
    # '/home/caozy/project/data/ECG_data/new_all_data/data7classes'
    config.signal_lead = "1"
    if config.dataset_type == 'chapman':
        print("chapman")
        if config.signal_lead == "1":
            if config.train_or_test == True:
                if config.num_class == 4:
                    print("train - 44444")
                    file_path = '/home/caozy/project/data/ECG_data/new_all_data/data4classes'
                    cls_list = ['AFIB_UNION', 'GSVT_UNION', 'SB_UNION', 'SR_UNION']
                if config.num_class == 7:
                    print("train - 77777")
                    file_path = '/home/guofengyi/data/chapman/data7classes'
                    cls_list = ['AFIB', 'SI', 'AF', 'SR', 'ST', 'SVT', 'SB']
                if config.num_class == 11:
                    print("train - 11111111")
                    file_path = '/home/guofengyi/data/chapman/data7classes'
                    # cls_list=['AFIB', 'SI', 'SR', 'ST', 'SVT', 'SB', 'AF','AT','AVNRT','AVRT','SAAWR']
                    cls_list = ['AFIB', 'SI', 'AF', 'SR', 'ST', 'SVT', 'SB', 'AT', 'AVNRT', 'AVRT', 'SAAWR']
            else:
                if config.num_class == 4:
                    print("test - 44444")
                    file_path = '/home/caozy/project/data/ECG_data/new_all_data/data4classes'
                    cls_list = ['AFIB_UNION', 'GSVT_UNION', 'SB_UNION', 'SR_UNION']
                # if config.num_class == 7:
                #     print("test - 77777")
                #     file_path = '/home/guofengyi/data/chapman/data7classes'
                #     cls_list = ['AFIB', 'SI', 'SR', 'ST', 'SVT', 'SB', 'AF']
                if config.num_class == 7:
                    print("test - few 4")
                    file_path = '/home/guofengyi/data/chapman/12_lead_ECG_new/few_data_classes'
                    cls_list = ['AT', 'AVNRT', 'AVRT', 'SAAWR']
        elif config.signal_lead == "12":
            file_path = '/home/guofengyi/data/chapman/12_lead_ECG_new_all_Lead/data7classes'
            cls_list = ['AFIB', 'SI', 'AF', 'SR', 'ST', 'SVT', 'SB']
        elif config.signal_lead == "1_2dim":
            file_path = '/home/caozy/project/data/ECG_data/12_lead_ECG_new_all_Lead/RP_data10s_7classes_10588'

        cls_list = ['AFIB', 'SI', 'AF', 'SR', 'ST', 'SVT', 'SB']
        id = 0
        data = []
        label = []

        for cls in cls_list:
            print("*" * 20 + f"load id{id} data" + "*" * 20)
            data_path = os.path.join(file_path, cls)
            # print(data_path)
            num = 0
            total_files = 0

            # 首先计算每个类别中文件的总数
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if os.path.splitext(file)[-1] == '.npy':
                        total_files += 1

            # 计算需要选择的样本数量（20%）
            num_samples = int(total_files *1)

            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if num >= num_samples:  # 如果已经达到20%的样本数量，就跳出循环
                        break
                    if os.path.splitext(file)[-1] == '.npy':
                        temp_data = np.load(os.path.join(root, file))
                        # print(os.path.join(root,file))
                        # print(temp_data.shape)

                        # 检查样本是否全零
                        if np.all(temp_data == 0):
                            print(f"Found all-zero sample in file: {data_path}")
                            pass
                        else:
                            data.append(temp_data)
                            label.append(id)
                            num += 1
            # data = z_score_normalize(data)

            print(f"{cls} id{id} 共{num}个数据")
            id += 1
        # return data, label

        cls_list_few = ['AT', 'AVNRT']
        file_path_few = '/home/guofengyi/data/chapman/data7classes'
        id = 7
        data_few = []
        label_few = []

        for cls in cls_list_few:
            print("*" * 20 + f"load id{id} data" + "*" * 20)
            data_path = os.path.join(file_path_few, cls)
            # print(data_path)
            num = 0
            total_files = 0

            # 首先计算每个类别中文件的总数
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if os.path.splitext(file)[-1] == '.npy':
                        total_files += 1

            # 计算需要选择的样本数量（20%）
            num_samples = int(total_files * 1)

            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if num >= num_samples:  # 如果已经达到20%的样本数量，就跳出循环
                        break
                    if os.path.splitext(file)[-1] == '.npy':
                        temp_data = np.load(os.path.join(root, file))
                        # print(os.path.join(root,file))
                        # print(temp_data.shape)

                        # 检查样本是否全零
                        if np.all(temp_data == 0):
                            print(f"Found all-zero sample in file: {data_path}")
                            pass
                        else:
                            data_few.append(temp_data)
                            label_few.append(id)
                            num += 1
            # data = z_score_normalize(data)

            print(f"{cls} id{id} 共{num}个数据")
            id += 1



    elif config.dataset_type == 'PTB':
        config.num_class = 12
        print("PTB")
        if config.num_class == 12:
            train_data_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/All_class_X_II_lead_origin_train.npy'
            train_label_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/All_class_II_T_onehot_train.npy'
            valid_data_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/All_class_X_II_lead_origin_valid.npy'
            valid_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/All_class_II_T_onehot_valid.npy'
            test_data_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/All_class_X_II_lead_origin_test.npy'
            test_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/All_class_II_T_onehot_test.npy'
        elif config.num_class == 6:
            train_data_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/X_train_filtered_most.npy'
            train_label_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/y_train_filtered_most.npy'
            test_data_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/X_test_filtered_most.npy'
            test_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/y_test_filtered_most.npy'
        elif config.num_class == 7:
            train_data_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/X_train_filtered_most_7.npy'
            train_label_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/y_train_filtered_most_7.npy'
            test_data_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/X_test_filtered_most_7.npy'
            test_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/y_test_filtered_most_7.npy'
        elif config.num_class == 4:
            train_data_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/X_train_filtered_few_4.npy'
            train_label_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/y_train_filtered_few_4.npy'
            test_data_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/X_test_filtered_few_4.npy'
            test_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/y_test_filtered_few_4.npy'

        # 读取文件
        train_data = np.load(train_data_filename)
        train_label = np.load(train_label_filename)
        test_data = np.load(test_data_filename)
        test_label = np.load(test_filename)
        valid_data = np.load(valid_data_filename)
        valid_label = np.load(valid_filename)
        data_array = np.concatenate((train_data, valid_data,test_data), axis=0)
        label = np.concatenate((train_label, valid_label,test_label), axis=0)

        # if config.num_samples != 0:
        #     print("删减是多标签的数据")
        #     train_data, train_label = filter_samples_by_label(train_data, train_label, target_label=0,
        #                                                       num_samples=config.num_samples, seed=42)
        #     test_data, test_label = filter_samples_by_label(test_data, test_label, target_label=0,
        #                                                     num_samples=int(config.num_samples / 10), seed=42)
        # else:
        #     print("不删减")
        # print("train_data.shape", train_data.shape)
        # print("train_label.shape", train_label.shape)
        # print("test_data.shape", test_data.shape)
        # print("test_label.shape", test_label.shape)

        def print_class_distribution(y, dataset_name):
            unique, counts = np.unique(y, return_counts=True)
            print(f"{dataset_name} 的类别分布数量:")
            for label, count in zip(unique, counts):
                print(f"类别 {label}: {count} 个样本")
            print()



        print("*" * 50)
        # data_array = np.concatenate((train_data,test_data),axis=0)
        data_array = score_z_norm(data_array)

        # label_array = np.concatenate((train_label, test_label), axis=0)
        # label =label.tolist()
        single_value_labels = np.argmax(label, axis=1)
        print_class_distribution(single_value_labels, f"（样本的类别）")
        few_target_values = {8, 7}
        few_indices = [index for index, value in enumerate(single_value_labels) if value in few_target_values]
        many_target_values = {0,1,2,3,4,5,6}
        many_indices = [index for index, value in enumerate(single_value_labels) if value in many_target_values]

        # for i in range(len(data_array[:, 0, 0])):
        #     data.append(data_array[i, :, :])
        data = data_array[many_indices]
        label = single_value_labels[many_indices]
        data_few = data_array[few_indices]
        label_few = single_value_labels[few_indices]
    return np.array(data), np.array(label),np.array(data_few), np.array(label_few)
        # return train_data, test_data, train_label, test_label




# def load_pic_data(config):
#     return train_dataset, val_dataset

def score_z_norm(data):
    # 计算每个特征的均值和标准差
    mean = np.mean(data, axis=(0, 1), keepdims=True)  # 计算均值
    std = np.std(data, axis=(0, 1), keepdims=True)  # 计算标准差
    std = np.where(std == 0, 1, std)  # 避免标准差为零的情况
    # 进行Z-score归一化
    newData = (data - mean) / std
    return newData

def window_slice(x, reduce_ratio=0.4):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    x = np.asarray(x)
    target_len = np.ceil(reduce_ratio * x.shape[1]).astype(int)
    print(f"target_len {target_len}")
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(low=0, high=x.shape[1] - target_len, size=(x.shape[0])).astype(int)
    print(f"starts {starts}")
    ends = (target_len + starts).astype(int)
    print(f"ends {ends}")

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i, :, dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len),
                                       pat[starts[i]:ends[i], dim]).T
    # print(f"np.asarray(ret).shape {np.asarray(ret).shape}")
    return ret







# 滑动平均滤波
def np_move_avg(a, n, mode="same"):
    return (np.convolve(a, np.ones((n,)) / n, mode=mode))


# 去除基线漂移
def baseline(ecg_filtered):
    b, a = signal.butter(8, 0.01, 'highpass')
    baseline = signal.filtfilt(b, a, ecg_filtered)
    # diff = ecg_filtered - baseline
    # plt.figure()
    # plt.subplot(311)
    # plt.plot(ecg_filtered[:1000])
    # plt.subplot(312)
    # plt.plot(baseline[:1000])
    # plt.subplot(313)
    # plt.plot(diff[:1000])
    # plt.show()
    return baseline


# 去除工频干扰
def filter(X):
    # 陷波处理滤除工频干扰
    fs = 500  # 采样频率
    f0 = 50.0  # 要去除的工频干扰
    Q = 30.0  # 品质因数
    b, a = signal.iirnotch(f0, Q, fs)
    X = signal.filtfilt(b, a, X)
    return X


# 滤波函数
def lvbo(X):
    X = np.array(X)
    # print()
    channels = 1
    X_smooth = np.random.rand(len(X[:, 0, 0]), 5000, 1)
    for index in range(len(X[:, 0, 0])):
        for channel in range(channels):
            X_smooth[index][:, channel] = filter(X[index][:, channel])  # X[index][:, channel]是ecg-original去除所有导联噪声
            X_smooth[index][:, channel] = np_move_avg(X_smooth[index][:, channel],
                                                      10)  # X[index][:, channel]是ecg-original
            X_smooth[index][:, channel] = baseline(X_smooth[index][:, channel])
    return X_smooth


def dimcon(x):
    t = []
    x_pin = torch.tensor(t)
    x = torch.tensor(x)
    print(x.shape)
    for i in range(12):
        x_er = x[:, :, i]
        x_era = preprocessing.scale(x_er)  # z分数归一化
        x_era = torch.Tensor(x_era)
        x_era = Variable(x_era)
        x_era = x_era.unsqueeze(1)  # 将x的维度进行扩展
        x_pin = torch.cat((x_pin, x_era), dim=1, out=None)
    return x_pin


if __name__ == "__main__":
    from single_lead_config import get_train_config
    config = get_train_config()
    config.dataset_type = 'chapman'
    train_data,train_label= load_data(config)
    train_data_array = np.array(train_data)
    train_label_array = np.array(train_label)
    # print(data.shape)
    data_tensor_train = dimcon(train_data_array)
    data_tensor_train_array= np.array(data_tensor_train)