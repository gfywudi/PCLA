import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchviz import make_dot
import torch.fft as fft
import neurokit2 as nk
import scipy
import random
import torchvision.transforms as transforms

import numpy as np

# class PrototypicalNetwork(nn.Module):
#     def __init__(self, encoder):
#         super(PrototypicalNetwork, self).__init__()
#         self.encoder = encoder

#     def forward(self, support, support_ys, query):
#         # 编码支持集和查询集
#         support = self.encoder(support)
#         query = self.encoder(query)

#         # 计算每个类别的原型
#         n_classes = support_ys.max() + 1
#         n_support = support.size(0) // n_classes
#         prototypes = support.view(n_classes, n_support, -1).mean(1)

#         # 计算查询样本到每个原型的距离
#         dists = torch.cdist(query, prototypes)

#         # 返回距离，用于计算损失和准确率
#         return -dists

#     def loss(self, dists, query_ys):
#         # 计算损失
#         return F.cross_entropy(-dists, query_ys)

# class PrototypicalNetwork(nn.Module):
#     def __init__(self, encoder):
#         super(PrototypicalNetwork, self).__init__()
#         self.encoder = encoder

#     def forward(self, support_set, support_labels, query_set):
#         """
#         Args:
#             support_set: Tensor of shape [num_support_samples, ...] - the support set.
#             support_labels: Tensor of shape [num_support_samples] - labels of the support set.
#             query_set: Tensor of shape [num_query_samples, ...] - the query set.

#         Returns:
#             dists: Tensor of shape [num_query_samples, num_classes] - distances from query samples to prototypes.
#         """
#         # Encode the support and query sets
#         support_embeddings = self.encoder(support_set)
#         query_embeddings = self.encoder(query_set)

#         # Calculate the prototypes
#         unique_labels = torch.unique(support_labels)
#         num_classes = len(unique_labels)
#         prototypes = torch.zeros((num_classes, support_embeddings.size(1))).to(support_embeddings.device)
#         for i, label in enumerate(unique_labels):
#             mask = (support_labels == label)
#             prototypes[i] = support_embeddings[mask].mean(dim=0)

#         # Calculate the distances from the query samples to the prototypes
#         dists = torch.cdist(query_embeddings, prototypes, p=2)  # Euclidean distance

#         return dists

#     def loss(self, dists, query_labels):
#         """
#         Args:
#             dists: Tensor of shape [num_query_samples, num_classes] - distances from query samples to prototypes.
#             query_labels: Tensor of shape [num_query_samples] - labels of the query set.

#         Returns:
#             loss: Scalar tensor - the loss value.
#         """
#         log_p_y = F.log_softmax(-dists, dim=1)
#         loss = F.nll_loss(log_p_y, query_labels)
#         return loss


class PrototypicalNetworkCosine(nn.Module):
    def __init__(self, encoder):
        super(PrototypicalNetworkCosine, self).__init__()
        self.encoder = encoder

    def forward(self, support_set, support_labels, query_set):
        # Encode the support and query sets
        support_embeddings = self.encoder(support_set, include_fc=False)
        query_embeddings = self.encoder(query_set, include_fc=False)

        # Normalize the embeddings so that we can calculate cosine similarity
        support_embeddings = F.normalize(support_embeddings, p=2, dim=1)
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

        # Calculate the prototypes
        unique_labels = torch.unique(support_labels)
        num_classes = len(unique_labels)
        prototypes = torch.zeros((num_classes, support_embeddings.size(1))).to(support_embeddings.device)
        for i, label in enumerate(unique_labels):
            mask = (support_labels == label)
            prototypes[i] = support_embeddings[mask].mean(dim=0)

        # Calculate the cosine similarity from the query samples to the prototypes
        # similarities = torch.mm(query_embeddings, prototypes.t())#余弦距离

        diff = query_embeddings.unsqueeze(1) - prototypes.unsqueeze(0)#欧氏距离
        squared_diff = diff ** 2
        similarities = torch.sqrt(squared_diff.sum(dim=2))
        return similarities

    def loss(self, similarities, query_labels):
        # We use log_softmax and nll_loss to calculate the cross-entropy loss
        log_p_y = F.log_softmax(similarities, dim=1)
        loss = F.nll_loss(log_p_y, query_labels)
        return loss


class CLloss(nn.Module):
    def __init__(self,device):
        super(CLloss, self).__init__()
        self.t = 0.1 #t是温度
        self.device = device
        self.cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)

    def forward(self, embed, embed_enhance, labels):
        labels = labels.float()
        embed = nn.functional.normalize(embed, dim=-1)
        batch_size = embed.size(0)
        t = self.t
        loss = torch.tensor(0.0).to(self.device)
        # print("labels",labels)
        # print("labels[0]", labels[0])
        # print("batch_size",batch_size)
        for i in range(batch_size):
            Ci = 1e-12
            Ei = 1e-12
            Li = 0.0
            for j in range(batch_size):
                if i == j:
                    continue
                else:
                    # print("labels[i]", labels[0])
                    Ci += torch.matmul(labels[i].unsqueeze(0), labels[j].unsqueeze(0))
                    # Ei += torch.dot(embed[i], embed_enhance[j])
                    # Ei += torch.exp(-nn.PairwiseDistance(p=2)(embed[i], embed[j]) / t)
                    Ei += torch.exp(-nn.CosineSimilarity(dim=0, eps=1e-6)(embed[i], embed_enhance[j]) / t)

            for j in range(batch_size):
                if i == j:
                    continue
                else:
                    Cij = torch.matmul(labels[i].unsqueeze(0), labels[j].unsqueeze(0)) / Ci
                    # Eij = -Cij * torch.log(torch.exp(-nn.PairwiseDistance(p=2)(embed[i], embed[j]) / t) / Ei)
                    Eij = -Cij * torch.log(torch.exp(-nn.CosineSimilarity(dim=0, eps=1e-6)(embed[i], embed_enhance[j]) / t) / Ei)
                    # Eij = -Cij * torch.log((torch.dot(embed[i], embed_enhance[j]) / t) / Ei)
                    Li += Eij

            loss += Li / batch_size

            return loss

class PrototypicalNetwork(nn.Module):
    def __init__(self, encoder, n_ways, n_shots, n_querys,device):
        super(PrototypicalNetwork, self).__init__()
        self.encoder = encoder
        self.n_ways = n_ways
        self.n_shots = n_shots
        self.n_querys = n_querys
        self.ACLoss = CLloss(device=device)

    def cl_loss(self,support_,query,support):
        support_ = support_.cpu().numpy()
        crop_resize = ECGCropResize(n=7, default_len=5000, fs=500)
        # crop_resize = ECGCycleMask(rate=0.5, fs=500)
        # crop_resize = ECGFrequencyDropOut(rate=0.3, default_len=5000)
        # crop_resize=AddGaussianNoise(noise_level=0.1, num_channels=1)
        processed_data = crop_resize(support_)

        enhence = self.encoder(processed_data, include_fc=False)

        # print("support enco", support.shape)
        loss_func = ContrastiveLoss(batch_size=enhence.shape[0])
        cl = loss_func(support, enhence)
        return cl,enhence

    def query_logist(self,support,query):
        nc = support.size(-1)
        support = support.view(-1, 1, self.n_ways, self.n_shots, nc)
        support = support.mean(axis=3)
        batch_size = support.size(0)
        query = query.view(batch_size, -1, 1, nc)
        logits = - ((query - support) ** 2).sum(-1)
        return logits

    def forward(self, support_, support_ys, query):
        # 编码支持集和查询集
        support = self.encoder(support_, include_fc=False)
        query = self.encoder(query, include_fc=False)

        # crop_resize = ECGFrequencyDropOut(rate=0.3, default_len=5000)
        # crop_resize = ECGCropResize(n=7, default_len=5000, fs=500)
        # processed_data = crop_resize(support_)
        # enhence = self.encoder(processed_data,include_fc=False)
        # # print("support enco", support.shape)
        # loss_func = ContrastiveLoss(batch_size=enhence.shape[0])
        # cl = loss_func(support,enhence)
        cl,enhence = self.cl_loss(support_, query, support)
        ACL_aug = self.ACLoss(support,enhence,support_ys)
        ACL_noaug = self.ACLoss(support, support, support_ys)

        nc = support.size(-1)
        support = support.view(-1, 1, self.n_ways, self.n_shots, nc)
        support = support.mean(axis=3)
        batch_size = support.size(0)
        query = query.view(batch_size, -1, 1, nc)

        logits = - ((query - support) ** 2).sum(-1)

        logits_enhance = self.query_logist(enhence,query)

        # print("lo", logits.shape) #torch.Size([1, 20, 4]) (batch,nway*query,  xx)
        return logits,logits_enhance,ACL_aug+ACL_noaug+cl#CL是无监督对比损失,ACL是初始原型的对比损失

    def loss(self, dists, query_ys):
        # print("query_ys",query_ys.shape)  #torch.Size([20])
        # 计算损失
        return F.cross_entropy(dists.view(-1, self.n_ways), query_ys)




class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))  # 超参数 温度
        self.register_buffer("negatives_mask", (
            ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())  # 主对角线为0，其余位置全为1的mask矩阵

    def forward(self, emb_i, emb_j):  # emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        z_i = F.normalize(emb_i, dim=1)  # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)  # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                                dim=2)  # simi_mat: (2*bs, 2*bs)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)  # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)  # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs

        nominator = torch.exp(positives / self.temperature)  # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)  # 2*bs, 2*bs

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss

class ECGFrequencyDropOut(object):#频率随机丢弃
    def __init__(self, rate=0.3, default_len=5000):
        self.rate = rate
        self.default_len = default_len
        self.num_zeros = int(self.rate * self.default_len)

    def __call__(self, data):
        num_zeros = random.randint(0, self.num_zeros)
        zero_idxs = sorted(np.random.choice(np.arange(self.default_len), num_zeros, replace=False))
        data_dct = scipy.fft.dct(data.copy())
        data_dct[:,:, zero_idxs] = 0
        data_idct = scipy.fft.idct(data_dct)

        return torch.tensor(data_idct).to('cuda')


class AddGaussianNoise(object):
    def __init__(self, noise_level=0.1, num_channels=1):
        self.noise_level = noise_level
        self.num_channels = num_channels

    def __call__(self, data):
        # 确保数据的形状是 (batchsize, 1, 5000)
        if data.ndim != 3 or data.shape[1] != 1 or data.shape[2] != 5000:
            raise ValueError("输入数据的形状必须是 (batchsize, 1, 5000)")

        # 生成与输入数据相同形状的噪声
        noise = np.random.normal(loc=0, scale=self.noise_level, size=data.shape)

        # 将噪声加到数据上
        noisy_data = data + noise

        return torch.tensor(noisy_data).to('cuda').float()





# Randomly select signals longer than n seconds and adjust them to 15s
class ECGCropResize(object):
    def __init__(self, n=7, default_len=5000, fs=500):
        self.min_len = n * fs
        self.default_len = default_len

    def __call__(self, data):
        crop_len = random.randint(self.min_len, self.default_len)
        crop_start = random.randint(0, self.default_len - crop_len)
        data_crop = data[:, 0,crop_start:crop_start + crop_len]
        data_resize = np.empty_like(data)
        x = np.linspace(0, crop_len-1, crop_len)
        xnew = np.linspace(0, crop_len-1, self.default_len)
        for i in range(data.shape[0]):
            f = scipy.interpolate.interp1d(x, data_crop[i], kind='cubic')
            data_resize[i] = f(xnew)

        return torch.tensor(data_resize).to('cuda')

# Select a certain length signal in each heartbeat cycle and set it to zero
class ECGCycleMask(object):
    def __init__(self, rate=0.5, fs=500):
        self.rate = rate
        self.fs = fs

    def __call__(self, data):

            # Extract R-peaks locations
        _, rpeaks = nk.ecg_peaks(np.float32(data[0,0,:]), sampling_rate=self.fs)
        r_peaks = rpeaks['ECG_R_Peaks']
        if len(r_peaks) > 1:
            cycle_len = int(np.mean(np.diff(r_peaks)))
            cut_len = int(self.rate * cycle_len)
            cut_start = random.randint(0, cycle_len - cut_len)
            data_ = data.copy()
            for r_idx in r_peaks:
                data_[:, :,r_idx + cut_start:r_idx + cut_start + cut_len] = 0
            return torch.tensor(data_).to('cuda')
        else:
            return torch.tensor(data).to('cuda')

# Randomly select less than the number of masks and set these channels to zero#不用这个
class ECGChannelMask(object):
    def __init__(self, masks=6, default_channels=12):
        self.masks = masks
        self.channels = np.arange(default_channels)

    def __call__(self, data):
        masks = random.randint(1, self.masks)
        channels_mask = np.random.choice(self.channels, masks, replace=False)
        data_ = data.copy()
        for channel_mask in channels_mask:
            data_[channel_mask] = 0
        return data_

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

if __name__ == "__main__":
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    # resnet = ResNet1D(ResNet1DBlock, [2, 2, 2, 2], num_classes=2)
    # transformer = TransformerEncoder(input_dim=512, num_heads=8, dim_feedforward=2048, num_layers=6)
    # model = ResNet1DTransformer(resnet, transformer, num_classes=2)
    # print(model)
    from cao_routine_training.RSTAnet import ResNet1D, ResNet1DBlock

    # 测试模型输入输出
    # 示例使用
    support_set = torch.randn(10, 1, 5000).to("cuda")  # 支持集样本
    support_labels = torch.randint(0, 2, (32,)).to("cuda")  # 支持集标签
    query_set = torch.randn(10, 1, 5000).to("cuda")  # 查询集样本
    ecg_network = ResNet1D(ResNet1DBlock, [2, 2, 2, 2, 2, 2, 2], num_classes=2).to("cuda")
    proto_network = PrototypicalNetwork(encoder=ecg_network,n_ways=2, n_shots=5,
                                n_querys=2, device="cuda:0")
    dists,dists_enhance,enhence = proto_network(support_set, support_labels, query_set)
    print("Output shape:", dists.shape)

    for k in proto_network.state_dict():
        print(k)

