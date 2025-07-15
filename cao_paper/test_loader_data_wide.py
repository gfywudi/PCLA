import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class MetaLearningDataset(Dataset):
    def __init__(self, data, n_ways, n_shots, n_queries,num_tasks, transform=None):
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

        # 生成所有任务的数据
        for _ in range(num_tasks):
            support_set, support_labels, query_set, query_labels = self._generate_task()
            self.tasks.append((support_set, support_labels, query_set, query_labels))
            # print(self.tasks)
        print("tasks:",len(self.tasks))
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
        # 在选定的类别中随机选择 n_ways 个类别
        selected_classes = np.random.choice(self.classes, self.n_ways, replace=False)
        # 打乱选定的类别顺序以增加随机性
        np.random.shuffle(selected_classes)

        support_set = []
        support_labels = []
        query_set = []
        query_labels = []

        for class_idx, cls in enumerate(selected_classes):
            # 从该类别中随机选择 n_shots + n_queries 个样本
            samples_id = np.random.choice(len(self.data[cls]), self.n_shots + self.n_queries, replace=False)

            np.random.shuffle(samples_id)

            task_data = [(self.data[cls][i], class_idx) for i in samples_id]

            # 分配支持集和查询集及其标签
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

        # 转换列表为张量
        support_set = torch.stack(support_set)
        support_labels = torch.tensor(support_labels)
        query_set = torch.stack(query_set)
        query_labels = torch.tensor(query_labels)

        return support_set, support_labels, query_set, query_labels

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


def score_z_norm(data):
    newData = np.zeros((data.shape))
    for i in range(12):
        mean = np.mean(data[:, :, i:i+1], axis=(0, 1), keepdims=True)
        std = np.std(data[:, :, i:i+1], axis=(0, 1), keepdims=True)
        std = np.where(std == 0, 1, std)
        newData[:, :, i] = (data[:, :, i] - mean) / std
    return newData


def z_score_normalize(data):
    """
    对数据进行 Z 分数归一化。
    :param data: List of NumPy arrays to be normalized.
    :return: List of normalized NumPy arrays.
    """
    # 将数据转换为 NumPy 数组，并将所有样本合并
    all_data = np.concatenate([d.flatten() for d in data])

    # 计算均值和标准差
    mean = np.mean(all_data)
    std = np.std(all_data)

    # 防止标准差为零
    if std == 0:
        std = 1

    # 对每个样本进行 Z 分数归一化
    normalized_data = [(d - mean) / std for d in data]

    return normalized_data

def get_test_dataset(configs):

    # n_ways, n_shots, n_queries, num_tasks
    n_ways=configs.n_way
    n_shots=configs.k_shot
    n_queries = configs.n_query
    # train_episode=configs.train_episode
    num_tasks=configs.test_episode
    # 定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将数据转换为张量
    ])
    # configs.dataset_type = 'chapman'
    configs.num_class = 4
    configs.num_samples= 0
    data = None
    # test_data = "PTB"
    # if test_data == 'chapman':
    if configs.dataset_type_test == 'chapman':
        # 定义数据根目录路径
        data_root = '/home/guofengyi/data/chapman/12_lead_ECG_new/few_data_classes'

        # 遍历数据根目录中的子文件夹（每个子文件夹对应一个类别）
        data = {}

        for class_folder in os.listdir(data_root):
            class_path = os.path.join(data_root, class_folder)
            if os.path.isdir(class_path):
                # 加载当前类别文件夹中的所有.npy文件
                class_data = []
                for filename in os.listdir(class_path):
                    if filename.endswith('.npy'):
                        file_path = os.path.join(class_path, filename)
                        # 使用NumPy加载.npy文件
                        sample = np.load(file_path)
                        # 应用转换
                        sample = transform(sample)
                        class_data.append(sample)
                # 将当前类别的数据列表添加到数据字典中
                class_data_normalized = z_score_normalize(class_data)
                data[class_folder] = class_data_normalized

        for class_folder, samples_list in data.items():
            print(f"train类别 {class_folder} 的样本数量:", len(samples_list))

        print(f"n_ways:{n_ways}, n_shots:{n_shots}, n_queries:{n_queries}, num_tasks:{num_tasks}")

    elif configs.dataset_type_test == 'PTB':
    # elif test_data == 'PTB':
        configs.num_class = 4
        print("PTB")
        if configs.num_class == 7:
            train_data_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/cao_data/X_origin_train_small.npy'
            train_label_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/cao_data/Y_onehot_train_small.npy'
            valid_data_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/cao_data/X_origin_valid_small.npy'
            valid_label_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/cao_data/Y_onehot_valid_small.npy'
            test_data_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/cao_data/X_origin_test_small.npy'
            test_label_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/cao_data/Y_onehot_test_small.npy'
        elif configs.num_class == 4:
            train_data_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/cao_data/X_origin_train_small.npy'
            train_label_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/cao_data/Y_onehot_train_small.npy'
            valid_data_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/cao_data/X_origin_valid_small.npy'
            valid_label_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/cao_data/Y_onehot_valid_small.npy'
            test_data_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/cao_data/X_origin_test_small.npy'
            test_label_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/cao_data/Y_onehot_test_small.npy'
        else:
            train_data_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/cao_data/X_origin_train_big.npy'
            train_label_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/cao_data/Y_onehot_train_big.npy'
            valid_data_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/cao_data/X_origin_valid_big.npy'
            valid_label_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/cao_data/Y_onehot_valid_big.npy'
            test_data_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/cao_data/X_origin_test_big.npy'
            test_label_filename = '/homeb/guofengyi/data/PTB_XL_rhythm/cao_data/Y_onehot_test_big.npy'
        # 读取文件
        train_data = np.load(train_data_filename)
        train_label = np.load(train_label_filename)
        test_data = np.load(test_data_filename)
        test_label = np.load(test_label_filename)
        valid_data = np.load(valid_data_filename)
        valid_label = np.load(valid_label_filename)
        data_wave = np.concatenate((train_data, valid_data,test_data), axis=0)
        label = np.concatenate((train_label, valid_label, test_label), axis=0)
        num_positive_labels = np.sum(label, axis=1)
        multilabel_indices = np.where(num_positive_labels > 1)[0]
        cleaned_labels = np.delete(label, multilabel_indices, axis=0)
        cleaned_data = np.delete(data_wave, multilabel_indices, axis=0)
        single_value_labels = np.argmax(cleaned_labels, axis=1)
        cleaned_data_Z_score = score_z_norm(cleaned_data)


        # if config.num_samples != 0:
        #     print("删减")
        #     cleaned_data, single_value_labels = filter_samples_by_label(cleaned_data, single_value_labels, target_label=0,
        #                                                       num_samples=config.num_samples, seed=42)
        #     # test_data, test_label = filter_samples_by_label(test_data, test_label, target_label=0,
        #     #                                                 num_samples=int(config.num_samples / 10), seed=42)
        # else:
        #     print("不删减")
        print("train_data.shape", cleaned_data.shape)
        print("train_label.shape", single_value_labels.shape)
        # print("test_data.shape", test_data.shape)
        # print("test_label.shape", test_label.shape)
        def print_class_distribution(y, dataset_name):
            unique, counts = np.unique(y, return_counts=True)
            print(f"{dataset_name} 的类别分布数量:")
            for label, count in zip(unique, counts):
                print(f"类别 {label}: {count} 个样本")
            # print()

        print_class_distribution(single_value_labels, f"训练集（样本的类别）")
        # print_class_distribution(test_label, f"测试集（样本的类别）")


        # return train_data, single_value_labels

        data_dict = {}
        label_list = ['SR','AFIB','STACH','SARRH','SBRAD','PACE', 'SVARR','BIGU','AFLT','SVTAC','PSVT','TRIGU']
        A= ['BIGU','AFLT','SVTAC','PSVT']
        #z归一化
        cleaned_data_Z_score = score_z_norm(cleaned_data)

        for j in range(len(label_list)):
            data_dict[label_list[j]] =[]
        for i in range(len(single_value_labels)):
            data_dict[label_list[single_value_labels[i]]].append(torch.tensor(cleaned_data_Z_score[i:i+1,:,1:2]))

        # data = {k: v for k, v in data_dict.items() if v}
        data = {k: v for k, v in data_dict.items() if k in A}

    # test_dataset_few = MetaLearningDataset(data, n_ways=n_ways, n_shots=n_shots, n_queries=n_queries, num_tasks=num_tasks, transform=None)
    #
    # data_list =[]
    # label_list =[]
    # # print("data",data)
    # a = -1
    # for key in data.keys():
    #     a+=1
    #     for ECG in data[key]:
    #         data_list.append(ECG)
    #         label_list.append(a)

    train_data_labelled = []
    label = -1
    for label_name, samples in data.items():
        print(label_name)
        label = label+1
        print(label)
        # print(samples)
        for sample in samples:
            train_data_labelled.append((label, sample))
    return train_data_labelled


if __name__ == "__main__":
    from single_lead_config import get_train_config

    config = get_train_config()
    # train_data, train_label = get_test_dataset(config)
    test_dataset_few = get_test_dataset(config)





