import torch.nn as nn
import torch.nn.functional as F
import torch


class MatchingNetwork(nn.Module):
    def __init__(self, encoder):
        """
        初始化 Matching Network
        :param input_dim: 输入特征的维度
        :param embedding_dim: 嵌入空间的维度
        """
        super(MatchingNetwork, self).__init__()
        # 定义嵌入层
        self.embedding = encoder

    def cosine_similarity(self, support_embeddings, query_embedding):
        """
        计算查询样本和支持样本之间的余弦相似度
        :param support_embeddings: 支持样本的嵌入 (N_support x embedding_dim)
        :param query_embedding: 查询样本的嵌入 (embedding_dim)
        :return: 支持样本和查询样本之间的相似度 (N_support)
        """
        support_norm = F.normalize(support_embeddings, p=2, dim=-1)
        query_norm = F.normalize(query_embedding, p=2, dim=-1)
        return torch.mm(support_norm, query_norm.unsqueeze(1)).squeeze(1)

    def forward(self, support_data, support_labels, query_data):
        """
        前向传播
        :param support_data: 支持样本数据 (N_support x input_dim)
        :param support_labels: 支持样本标签 (N_support)
        :param query_data: 查询样本数据 (N_query x input_dim)
        :return: 查询样本的分类概率 (N_query x N_classes)
        """
        # 获取支持样本和查询样本的嵌入
        support_embeddings = self.embedding(support_data)
        query_embeddings = self.embedding(query_data)

        # 获取支持样本的独特类别
        unique_labels = torch.unique(support_labels)
        N_classes = len(unique_labels)

        # 对每个查询样本计算与每个类别的相似度
        query_probs = []
        for query_embedding in query_embeddings:
            # 计算查询样本与支持样本之间的相似度
            similarities = self.cosine_similarity(support_embeddings, query_embedding)
            # 初始化类别概率
            class_probs = torch.zeros(N_classes, device=query_data.device)
            for i, label in enumerate(unique_labels):
                # 筛选出属于当前类别的支持样本
                label_mask = (support_labels == label)
                # 类别概率为当前类别支持样本相似度的和
                class_probs[i] = similarities[label_mask].sum()
            # 对类别概率进行归一化
            query_probs.append(F.softmax(class_probs, dim=0))
        return torch.stack(query_probs, dim=0)
