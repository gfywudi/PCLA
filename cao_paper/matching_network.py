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
# from train_loader_data_wide import get_train_dataset
# from test_loader_data_wide import get_test_dataset
from meta_test_loader_data import get_test_dataset
from meta_train_loader_data import get_train_dataset
from tqdm import tqdm



class DistanceNetwork(nn.Module):
    def __init__(self):
        super(DistanceNetwork, self).__init__()

    def forward(self, support_set, input_image):

        """
        Produces pdfs over the support set classes for the target set image.
        :param support_set: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, 64]
        :param input_image: The embedding of the target image, tensor of shape [batch_size, 64]
        :return: Softmax pdf. Tensor with cosine similarities of shape [batch_size, sequence_length]
        """
        eps = 1e-10
        similarities = []
        for support_image in support_set:
            # sum_support = torch.sum(torch.pow(support_image, 2), 1)
            support_magnitude = support_image.clamp(eps, float("inf")).rsqrt()
            dot_product = input_image.unsqueeze(0).bmm(support_image.unsqueeze(1)).squeeze()
            cosine_similarity = dot_product * support_magnitude
            similarities.append(cosine_similarity)
        similarities = torch.stack(similarities)
        return similarities

class AttentionalClassify(nn.Module):
    def __init__(self):
        super(AttentionalClassify, self).__init__()

    def forward(self, similarities, support_set_y):

        """
        Produces pdfs over the support set classes for the target set image.
        :param similarities: A tensor with cosine similarities of size [sequence_length, batch_size]
        :param support_set_y: A tensor with the one hot vectors of the targets for each support set image
                                                                            [sequence_length,  batch_size, num_classes]
        :return: Softmax pdf
        """
        softmax = nn.Softmax()
        softmax_similarities = softmax(similarities)
        preds = softmax_similarities.unsqueeze(1).bmm(support_set_y).squeeze()

        return preds

class MatchingNetwork(nn.Module):
    def __init__(self, network, keep_prob, \
                 batch_size=100, num_channels=1, learning_rate=0.001, fce=False, num_classes_per_set=5, \
                 num_samples_per_class=1, nClasses = 0, image_size = 28):
        super(MatchingNetwork, self).__init__()

        """
        Builds a matching network, the training and evaluation ops as well as data augmentation routines.
        :param keep_prob: A tf placeholder of type tf.float32 denotes the amount of dropout to be used
        :param batch_size: The batch size for the experiment
        :param num_channels: Number of channels of the images
        :param is_training: Flag indicating whether we are training or evaluating
        :param rotate_flag: Flag indicating whether to rotate the images
        :param fce: Flag indicating whether to use full context embeddings (i.e. apply an LSTM on the CNN embeddings)
        :param num_classes_per_set: Integer indicating the number of classes per set
        :param num_samples_per_class: Integer indicating the number of samples per class
        :param nClasses: total number of classes. It changes the output size of the classifier g with a final FC layer.
        :param image_input: size of the input image. It is needed in case we want to create the last FC classification 
        """
        self.batch_size = batch_size
        self.fce = fce
        self.g = network
        # if fce:
        #     self.lstm = BidirectionalLSTM(layer_sizes=[32], batch_size=self.batch_size, vector_dim = self.g.outSize)
        self.dn = DistanceNetwork()
        self.classify = AttentionalClassify()
        self.keep_prob = keep_prob
        self.num_classes_per_set = num_classes_per_set
        self.num_samples_per_class = num_samples_per_class
        self.learning_rate = learning_rate

    def forward(self, support_set_images, support_set_labels_one_hot, target_image, target_label):
        """
        Builds graph for Matching Networks, produces losses and summary statistics.
        :param support_set_images: A tensor containing the support set images [batch_size, sequence_size, n_channels, 28, 28]
        :param support_set_labels_one_hot: A tensor containing the support set labels [batch_size, sequence_size, n_classes]
        :param target_image: A tensor containing the target image (image to produce label for) [batch_size, n_channels, 28, 28]
        :param target_label: A tensor containing the target label [batch_size, 1]
        :return:
        """
        # produce embeddings for support set images
        # encoded_images = []
        # for i in np.arange(support_set_images.size(1)):
        support_set = self.g(support_set_images, include_fc=False)
            # encoded_images.append(gen_encode)


        # produce embeddings for target images
        for i in np.arange(target_image.size(0)):
            target_image_ = self.g(target_image[i:i+1,:,:], include_fc=False)
            # support_set.append(target_image_)
            # outputs = torch.stack(support_set)
                # encoded_images.append(gen_encode)
                # outputs = torch.stack(encoded_images)

                # if self.fce:
                #     outputs, hn, cn = self.lstm(outputs)

                # get similarity between support set embeddings and target
            similarities = self.dn(support_set=support_set, input_image=target_image_)
            similarities = similarities.t()

            # produce predictions for target probabilities
            preds = self.classify(similarities,support_set_y=support_set_labels_one_hot)

            # calculate accuracy and crossentropy loss
            values, indices = preds.max(1)
            if i == 0:
                accuracy = torch.mean((indices.squeeze() == target_label[:,i]).float())
                crossentropy_loss = F.cross_entropy(preds, target_label[:,i].long())
            else:
                accuracy = accuracy + torch.mean((indices.squeeze() == target_label[:, i]).float())
                crossentropy_loss = crossentropy_loss + F.cross_entropy(preds, target_label[:, i].long())

            # delete the last target image encoding of encoded_images
            support_set.pop()

        return accuracy/target_image.size(1), crossentropy_loss/target_image.size(1)





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
        # for i, data in loop:
        #     loop = tqdm(enumerate(final_train_loader), total=len(final_train_loader))
        for idx, batch in loop:
            # for idx, batch in enumerate(final_train_loader):
            optimizer.zero_grad()
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

            # 前向传播
            acc, loss = net(support_set, support_labels, query_set, query_labels)


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

    os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    wayshot_list = [[2, 1], [4, 1], [2, 5], [4, 5]]
    config = get_train_config()
    config.train_episode = 1000
    config.dataset_type_train = "chapman"
    config.dataset_type_test = "chapman"
    config.n_query = 1
    project_name = "siamese"
    batchsize =128
    epochs = 100
    for way_shot in wayshot_list:
        best_Score_record = []
        lr_rates = [0.001, 0.005, 0.0001, 0.0005]
        config.n_way = way_shot[0]
        config.k_shot = way_shot[1]
        criterion = ContrastiveLoss()
        train_data = get_train_dataset(config)
        # train_dataset = SiameseDataset(data=train_data,
        #                                transform=None,
        #                                should_invert=False)

        train_dataloader = DataLoader(train_data, batch_size=batchsize, num_workers=0, shuffle=True)

        # test_data = get_test_dataset(config)
        test_data,_,_ = get_test_dataset(config)
        # valid_dataset = SiameseDataset(data=test_data,
        #                                transform=None,
        #                                should_invert=False)
        valid_dataloader = DataLoader(test_data, batch_size=1, num_workers=0, shuffle=True)

        ecg_network = ResNet1D(ResNet1DBlock, [2, 2, 2, 2, 2, 2, 2], num_classes=7)
        matchingNet = MatchingNetwork(network=ecg_network, batch_size=100,
                                           keep_prob=0.5, num_channels=1,
                                           fce=False,
                                           num_classes_per_set=config.n_way,
                                           num_samples_per_class=config.k_shot,
                                           nClasses=0, image_size=28).cuda()
        # model = SiameseNetwork(encoder=ecg_network).cuda()
        print("begin training")
        for lr_rate in lr_rates:
            acc = train(matchingNet, train_dataloader, valid_dataloader, epochs, criterion,lr_rate)
            best_Score_record.append(acc)
        output_dir = f"/homeb/guofengyi/result/cao_paper/few_shot_result_txt"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # 打开文件以写入
        # output_name = f"resnet-senet-7block.txt  resnet-2n-8block"
        output_name = f"{config.dataset_type_train}_to{config.dataset_type_test}"+project_name+f"{config.n_way}way_{config.k_shot}shot.txt"
        output_file = os.path.join(output_dir, output_name)
        with open(output_file, "a") as file:
            file.write("-" * 20 + project_name + "-" * 20 + "\n")

            for lr_rate, best_score in zip(lr_rates, best_Score_record):
                # file.write(f"lr_rate: {lr_rate}  best_acc: {best_score[0]:.4f} best_recall: {best_score[1]:.4f}  best_precision: {best_score[2]:.4f}  best_f1: {best_score[3]:.4f}  best_roc_auc: {best_score[4]:.4f}\n")
                file.write(f"lr_rate: {lr_rate}  best_acc: {best_score:.4f}\n")
            file.write(project_name + "\n")
