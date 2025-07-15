import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    """
    孪生网络用于少样本分类，输入格式与原型网络一致
    """

    def __init__(self, encoder):
        super(SiameseNetwork, self).__init__()

        # 嵌入网络（用于提取特征）
        self.embedding_net = encoder

    def forward_one(self, x):
        """
        对单个输入进行特征嵌入
        """
        return self.embedding_net(x)

    def forward(self, support_set, query_set, support_labels):
        """
        对支持集和查询集分别进行嵌入特征提取，并输出查询集预测
        Args:
            support_set: (N, 1, H, W) 支持集样本
            query_set: (Q, 1, H, W) 查询集样本
            support_labels: (N, C) 支持集样本的 one-hot 标签

        Returns:
            predictions: (Q, C) 查询集的类别概率分布
        """
        # 提取支持集和查询集的嵌入特征
        support_embeddings = self.forward_one(support_set)  # (N, D)
        query_embeddings = self.forward_one(query_set)  # (Q, D)

        # 计算查询集与支持集之间的欧几里得距离
        distances = -torch.cdist(query_embeddings, support_embeddings, p=2)  # (Q, N)

        # 通过 softmax 计算注意力权重
        attention_weights = F.softmax(distances, dim=1)  # (Q, N)

        # 加权支持集标签，生成查询集的预测分布
        predictions = torch.matmul(attention_weights, support_labels)  # (Q, C)

        return predictions


# 示例代码




