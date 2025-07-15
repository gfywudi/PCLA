import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
import os
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from cao_routine_training.RSTAnet import ResNet1D,ResNet1DBlock
from train_loader_data_wide import get_train_dataset
# from test_loader_data_wide import get_test_dataset
from meta_test_loader_data import get_test_dataset
from tqdm import tqdm


class SiameseDataset(Dataset):
    def __init__(self,  data, transform=None, should_invert=True):
        self.imageFolderDataset = data
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        should_get_same_class = random.randint(0, 1)
        # print("imageFolderDataset:", self.imageFolderDataset)
        img1_tuple = random.choice(self.imageFolderDataset)


        if should_get_same_class:
            while True:
                img2_tuple = random.choice(self.imageFolderDataset)
                # print("img1_tuple[1]:", img1_tuple[0])
                # print("img2_tuple[1]:", img2_tuple[0])
                # print("img1_tuple[1].shape:", img1_tuple[0].shape)
                # print("img2_tuple[1].shape:", img2_tuple[0].shape)
                if img1_tuple[0] == img2_tuple[0]:
                    break
        else:
            while True:
                img2_tuple = random.choice(self.imageFolderDataset)
                # print("img1_tuple[1]:", img1_tuple[0])
                # print("img2_tuple[1]:", img2_tuple[0])
                # print("img1_tuple[1].shape:", img1_tuple[0].shape)
                # print("img2_tuple[1].shape:", img2_tuple[0].shape)
                if img1_tuple[0] != img2_tuple[0]:
                    break

        img1 = img1_tuple[1]
        img2 = img2_tuple[1]

        return img1, img2, torch.from_numpy(np.array([(img1_tuple[0] != img2_tuple[0])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset)

class OmniglotTask(object):
    # This class is for task generation for both meta training and meta testing.
    # For meta training, we use all 20 samples without valid set (empty here).
    # For meta testing, we use 1 or 5 shot samples for training, while using the same number of samples for validation.
    # If set num_samples = 20 and chracter_folders = metatrain_character_folders, we generate tasks for meta training
    # If set num_samples = 1 or 5 and chracter_folders = metatest_chracter_folders, we generate tasks for meta testing
    def __init__(self, character_folders, num_classes, train_num, test_num):
        """
         character_folders : ['../datas/omniglot_resized/Gujarati\\character19',...,'../datas/omniglot_resized/Greek\\character20']
         num_classes : 5   每个情境num_classes个类
         train_num :1  每类train_num张训练图
         test_num : 10 每类test_num个查询图
         """
        self.character_folders = character_folders
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num

        class_folders = random.sample(self.character_folders,self.num_classes)
        labels = np.array(range(len(class_folders)))
        labels = dict(zip(class_folders, labels))
        samples = dict()

        self.train_roots = []
        self.test_roots = []
        for c in class_folders:

            temp = [os.path.join(c, x) for x in os.listdir(c)]
            samples[c] = random.sample(temp, len(temp))

            self.train_roots += samples[c][:train_num]
            self.test_roots += samples[c][train_num:train_num+test_num]

        self.train_labels = [labels[self.get_class(x)] for x in self.train_roots]
        self.test_labels = [labels[self.get_class(x)] for x in self.test_roots]






class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(128,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out


# define contrastive loss
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


def train(net, train_dataloader, valid_dataloader, epochs, criterion,lr_rate):
    optimizer = optim.AdamW(net.parameters(), lr=lr_rate)
    train_loss = []  # training loss for every epoch
    valid_loss = []  # validation loss for every epoch
    sum_train_loss = 0.0  # sum of training losses for every epoch
    sum_valid_loss = 0.0  # sum of validation losses for every epoch
    final_acc = []
    for epoch in range(1, epochs + 1):
        train_epoch_loss = 0.0
        net.train()
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, data in loop:
        # for i, data in enumerate(train_dataloader, 0):
            img1, img2, label = data
            img1, img2, label = img1.squeeze(-1).float() .cuda(), img2.squeeze(-1).float() .cuda(), label.cuda()

            label = label.float()

            output1, output2 = net(img1, img2)
            loss = criterion(output1, output2, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_epoch_loss = train_epoch_loss + ((1 / (i + 1)) * (loss.item() - train_epoch_loss))
            loop.set_description(f"train Epoch: [{epochs}][{epoch}/{len(train_dataloader)}]\t")
            loop.set_postfix(trainloss_all=train_epoch_loss)
        train_loss.append(train_epoch_loss)
        sum_train_loss += train_epoch_loss
        print("testing")
        valid_epoch_loss = 0.0
        correct_count = 0
        correct = 0
        with torch.no_grad():
            net.eval()
            number = 0
            # loop = tqdm(enumerate(valid_dataloader), total=len(valid_dataloader))
            for i, data in enumerate(valid_dataloader, 0):
                support_x = data[
                    0].cuda()  # [batch_size, n_way*(k_shot+k_query), c , h , w] torch.Size([11, 5, 1, 5000, 1])
                support_x =support_x.squeeze(4)
                support_y = data[1].cuda()
                query_x = data[2].cuda().squeeze(4)
                query_y = data[3].cuda()
                for j in range(len(query_x[0,:,0,0])):
                    label_range_list = []
                    support_y_list = []
                    correct += 1
                    for i in range(len(support_x[0,:,0,0])):
                        img1 = support_x[:,i,:,:]
                        img2 = query_x[:,j,:,:]
                        output1, output2 = net(img1.float(), img2.float())
                        # loss = criterion(output1, output2, label)
                        euclidean_distance = F.pairwise_distance(output1, output2)
                        prediction = torch.sigmoid(euclidean_distance)
                        support_y_list.append(prediction.cpu().numpy())
                        label_range_list.append(prediction.cpu().numpy())
                    max_value = max(support_y_list)
                    max_index = support_y_list.index(max_value)
                    predict_label = support_y[:,max_index]
                    real_label = query_y[:,j]
                    if predict_label == real_label:
                        correct_count += 1
            acc = correct_count/correct

        #
        #         img1, img2, label = data
        #         img1, img2, label = img1.squeeze(-1).float() .cuda(), img2.squeeze(-1).float() .cuda(), label.cuda()
        #
        #         output1, output2 = net(img1, img2)
        #         loss = criterion(output1, output2, label)
        #
        #         valid_epoch_loss = valid_epoch_loss + ((1 / (i + 1)) * (loss.item() - valid_epoch_loss))
        #
        #         ######################
        #         #############################
        #         euclidean_distance = F.pairwise_distance(output1, output2)
        #         # print(output1,output2)
        #         prediction = torch.sigmoid(euclidean_distance)
        #         # print(prediction)
        #         total = label.size(0)
        #         # print("Testing...")
        #         # check if prediction and actual label are same
        #         for j in range(label.size(0)):
        #             if (prediction[j] > 0.5) and (label[j] == 1):
        #                 correct += 1
        #             elif (prediction[j] < 0.5) and (label[j] == 0):
        #                 correct += 1
        #         correct_count += correct / total
        #         correct = 0
        #         number += 1
        #         print(number)
        #         # print('Pred : {:.2f} Label : {}'.format(prediction.item(), label.item()))
        # valid_loss.append(valid_epoch_loss)
        # sum_valid_loss += valid_epoch_loss

        print("Epoch {}/{}\n Train loss : {} \t Valid loss {}\n acc{}\n"
              .format(epoch, epochs, train_epoch_loss, valid_epoch_loss, acc))
        # correct_count, count, acc = eval(net, valid_dataloader)

    # print("Average training loss after {} epochs : {}".format(epochs, sum_train_loss / epochs))
    # print("Average validation loss after {} epochs : {}".format(epochs, sum_valid_loss / epochs))

    return acc

def eval(net, test_dataloader):
    with torch.no_grad():
        net.eval()
        count = 100
        correct = 0
        correct_count = 0
        dataiter = iter(test_dataloader)
        print("Testing...")

        img1, img2, label = next(dataiter)

        # cat = torch.cat((img1.squeeze(-1), img2.squeeze(-1)), 0)
        output1, output2 = net(Variable(img1.squeeze(-1)).float().cuda(), Variable(img2.squeeze(-1)).float() .cuda())
        euclidean_distance = F.pairwise_distance(output1, output2)
        prediction = torch.sigmoid(euclidean_distance)
        total = label.size(0)
        print("Testing...")
        # check if prediction and actual label are same
        for j in range(label.size(0)):
            if (prediction[j] > 0.5) and (label[j] == 1):
                correct += 1
            elif (prediction[j] < 0.5) and (label[j] == 0):
                correct += 1

        correct_count += correct / total
        correct = 0
        print('Pred : {:.2f} Label : {}'.format(prediction.item(), label.item()))
    return correct_count, count, (correct_count / count) * 100

if __name__ == '__main__':
    from single_lead_config import get_train_config

    wayshot_list = [[2, 1], [4, 1], [2, 5], [4, 5]]
    config = get_train_config()
    config.dataset_type_train = "chapman"
    config.dataset_type_test = "chapman"
    config.n_query = 1
    project_name = "siamese"
    batchsize =128
    epochs = 100
    for way_shot in wayshot_list:
        best_Score_record = []
        lr_rates = [0.00000001, 0.00000005, 0.0000001, 0.0000005, 0.000001, 0.000005, 0.00001, 0.00005, 0.1, 0.5, 0.01,
                    0.05, 0.001, 0.005, 0.0001, 0.0005]
        config.n_way = way_shot[0]
        config.k_shot = way_shot[1]
        criterion = ContrastiveLoss()
        train_data = get_train_dataset(config)
        train_dataset = SiameseDataset(data=train_data,
                                       transform=None,
                                       should_invert=False)
        train_dataset = OmniglotTask(train_data, config.n_way, config.k_shot, config.n_query )
        train_dataloader = DataLoader(train_dataset, batch_size=batchsize, num_workers=0, shuffle=True)

        # test_data = get_test_dataset(config)
        test_data,_,_ = get_test_dataset(config)
        # valid_dataset = SiameseDataset(data=test_data,
        #                                transform=None,
        #                                should_invert=False)
        valid_dataloader = DataLoader(test_data, batch_size=1, num_workers=0, shuffle=True)

        ecg_network = ResNet1D(ResNet1DBlock, [2, 2, 2, 2, 2, 2, 2], num_classes=7)
        model = SiameseNetwork(encoder=ecg_network).cuda()
        print("begin training")
        for lr_rate in lr_rates:
            acc = train(model, train_dataloader, valid_dataloader, epochs, criterion,lr_rate)
            best_Score_record.append(acc)
        output_dir = f"/homeb/guofengyi/result/cao_paper/few_shot_result_txt"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # 打开文件以写入
        # output_name = f"resnet-senet-7block.txt  resnet-2n-8block"
        output_name = f"{config.dataset_type_train}_to{config.dataset_type_test}"+project_name+f"{config.n_way}way_{configs.k_shot}shot.txt"
        output_file = os.path.join(output_dir, output_name)
        with open(output_file, "a") as file:
            file.write("-" * 20 + project_name + "-" * 20 + "\n")

            for lr_rate, best_score in zip(lr_rates, best_Score_record):
                # file.write(f"lr_rate: {lr_rate}  best_acc: {best_score[0]:.4f} best_recall: {best_score[1]:.4f}  best_precision: {best_score[2]:.4f}  best_f1: {best_score[3]:.4f}  best_roc_auc: {best_score[4]:.4f}\n")
                file.write(f"lr_rate: {lr_rate}  best_acc: {best_score:.4f}\n")
            file.write(project_name + "\n")
