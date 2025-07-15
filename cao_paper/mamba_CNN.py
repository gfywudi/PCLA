import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mamba_ssm.modules.mamba_simple import Mamba, Block




class Model_routine(torch.nn.Module):
    def __init__(self, num_classes=15):
        super(Model_routine, self).__init__()

        # if self.configs.revin == 1:
        #     self.revin_layer = RevIN(self.configs.enc_in)
        self.ch_ind = 1
        self.residual = 1
        self.layer_num = 4
        self.conv_out = 20#625resnet
        self.input_dim = 256
        self.lead_number = 176#48resnet
        # self.conv1 = nn.Sequential(
        #     nn.Conv1d(in_channels=12, out_channels=12, kernel_size=10, padding=0,groups=12),
        #     nn.BatchNorm1d(num_features=12, eps=1e-05, momentum=0.1, affine=True),
        #     nn.LeakyReLU())
        self.lin1 = nn.Sequential(torch.nn.Linear(self.conv_out, self.input_dim),
                                   nn.BatchNorm1d(num_features=self.lead_number, eps=1e-05, momentum=0.1, affine=True),
                                   nn.LeakyReLU())
        self.dropout1 = torch.nn.Dropout(0.5)

        # self.lin2 =  nn.Sequential(torch.nn.Linear(256, 128),
        #                            nn.BatchNorm1d(num_features=12, eps=1e-05, momentum=0.1, affine=True),
        #                            nn.LeakyReLU())
        self.dropout2 = torch.nn.Dropout(0.5)
        self.dropout3 = torch.nn.Dropout(0.5)
        self.dropout4 = torch.nn.Dropout(0.4)
        if self.ch_ind == 1:
            self.d_model_param1 = 12
            self.d_model_param2 = 12

        else:
            self.d_model_param1 = 128
            self.d_model_param2 = 256


        self.mamba3 = nn.Sequential(nn.LayerNorm(self.input_dim),
            Mamba(d_model=self.input_dim, d_state=4, d_conv=2,
                            expand=1))
        self.mamba3_1 = nn.Sequential(nn.LayerNorm(self.input_dim),
            Mamba(d_model=self.input_dim, d_state=4, d_conv=2,
                            expand=1))
        self.mamba3_2 = nn.Sequential(nn.LayerNorm(self.input_dim),
                                      Mamba(d_model=self.input_dim, d_state=4, d_conv=2,
                                            expand=1))
        self.mamba3_3 = nn.Sequential(nn.LayerNorm(self.input_dim),
                                      Mamba(d_model=self.input_dim, d_state=4, d_conv=2,
                                            expand=1))

        self.mamba4 = nn.Sequential(nn.LayerNorm(self.lead_number),Mamba(d_model=self.lead_number, d_state=4, d_conv=2,
                            expand=1))
        self.mamba4_1 = nn.Sequential(nn.LayerNorm(self.lead_number),Mamba(d_model=self.lead_number, d_state=4, d_conv=2,
                            expand=1))
        self.mamba4_2 = nn.Sequential(nn.LayerNorm(self.lead_number), Mamba(d_model=self.lead_number, d_state=4, d_conv=2,
                                                              expand=1))
        self.mamba4_3 = nn.Sequential(nn.LayerNorm(self.lead_number), Mamba(d_model=self.lead_number, d_state=4, d_conv=2,
                                                              expand=1))

        self.fc3 = nn.Sequential(nn.Linear(16*self.lead_number, num_classes),
                                 nn.Sigmoid())
                                    # nn.Softmax())
        self.lin5 = nn.Sequential(nn.BatchNorm1d(num_features=self.lead_number, eps=1e-05, momentum=0.1, affine=True),
                                  torch.nn.Linear(self.input_dim, 128),
                                  torch.nn.Linear(128, 64),
                                torch.nn.Linear(64, 16))

        self.sk_conv1 = nn.Sequential(
            nn.Conv1d(12, 12, kernel_size=11),
            nn.BatchNorm1d(num_features=12, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.Linear(self.input_dim, self.input_dim),
        )

        self.sk_conv2 = nn.Sequential(
            nn.Conv1d(5000, 5000, kernel_size=1),
            nn.BatchNorm1d(num_features=5000, eps=1e-05, momentum=0.1, affine=True),
        )
        # self.resnet_encoder = ML_Resnet()
        # self.densnet_encoder = Densenet_mult_encoder()
        # self.CNN_encoder = CNN_encoder()
    def forward(self, x):


        # x = self.CNN_encoder(x)
        # x = self.densnet_encoder(x)
        # x = self.resnet_encoder(x)
        # x = self.batchnorm1(x)
        x = self.multi_scale(x)
        # x = self.resnet(x)
        x = self.lin1(x)
        x = self.dropout1(x)
        x = self.mamba3_1(x)
        # x = self.mamba3_2(x)
        # x = self.dropout1(x)

        x = self.lin5(x)
        out = x.view(x.shape[0], -1)
        out = self.fc3(out)
        # out = self.fc(out)



        return out


class ECGFeatureExtractor_mamba(nn.Module):
    def __init__(self, input_channels, cnn_output_channels, model_dim, num_heads, num_layers, dropout=0.1):
        super(ECGFeatureExtractor_mamba, self).__init__()
        self.input_dim = 64
        # 一维卷积层，用于提取时序特征
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32)  # 使用自适应平均池化将特征图大小变为1
        )

        self.input_linear = nn.Linear(2048, 4096)  # 调整卷积层输出的维度以匹配Transformer的嵌入维度
        # self.positional_encoding = PositionalEncoding(64, dropout)
        self.mamba3_1 = nn.Sequential(nn.LayerNorm(self.input_dim),
                                      Mamba(d_model=self.input_dim, d_state=4, d_conv=2,
                                            expand=1))
        self.mamba3_2 = nn.Sequential(nn.LayerNorm(self.input_dim),
                                      Mamba(d_model=self.input_dim, d_state=4, d_conv=2,
                                            expand=1))


        transformer_layer = nn.TransformerEncoderLayer(d_model=64, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

        # self.class_linear = nn.Linear(cnn_output_channels, 128)

    def forward(self, x):
        x = self.cnn(x)  # 通过CNN处理时序数据 torch.Size([110, 64, 625])
        # print("------cnn:",x.shape)  # cnn: torch.Size([20, 64, 625]))

        x = x.view(x.size(0), -1)  # 展平除了批次维度之外的所有维度
        # print("------After flattening:", x.shape)  # 查看展平后的形状

        x = self.input_linear(x).view(x.size(0), 64, 64)  # 调整特征维度  input_linear: torch.Size([110, 64, 128])
        # print("input_linear:",x.shape)

        x = self.mamba3_1(x)
        x = self.mamba3_2(x)




#########################################
        # x = x.permute(1, 0,
        #               2)  # 调整维度以符合Transformer的输入要求：(seq_len, batch, features) x.permute: torch.Size([64, 110, 128])
        # # print("x.permute:",x.shape)
        # # x = self.positional_encoding(x)
        # # print("positional_encoding:",x.shape)  # positional_encoding: torch.Size([64, 110, 128])
        # x = self.transformer(x)  # transformer: torch.Size([64, 110, 128])
        # # print("transformer:",x.shape)
 #################       #################
        return x


##############################new_rstanet
class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim=None, dropout=0.1):
        super(AttentionLayer, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

        self.feedforward = None
        if feedforward_dim is not None:
            # 可选的前馈网络
            self.feedforward = nn.Sequential(
                nn.Linear(embed_dim, feedforward_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(feedforward_dim, embed_dim),
            )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim) if self.feedforward is not None else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 多头注意力
        attn_output, _ = self.multihead_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        # 可选的前馈网络
        if self.feedforward is not None:
            ff_output = self.feedforward(x)
            x = self.norm2(x + self.dropout(ff_output))

        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResNet1DBlock(nn.Module):
    expansion = 1  # 将expansion定义为类变量而不是实例变量

    def __init__(self, in_channels, out_channels, stride):
        super(ResNet1DBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.se1 = SELayer(out_channels)  # 添加SE模块
        self.conv2 = nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels * self.expansion)
        self.se2 = SELayer(out_channels * self.expansion)  # 添加SE模块
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.se1(out)  # 应用SE模块
        out = self.bn2(self.conv2(out))
        out = self.se2(out)  # 应用SE模块
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet1D(nn.Module):
    def __init__(self, block, layers, num_classes=4):
        super(ResNet1D, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 256, layers[4], stride=2)
        self.layer6 = self._make_layer(block, 512, layers[5], stride=2)
        self.layer7 = self._make_layer(block, 1024, layers[6], stride=2)
        self.input_linear = nn.Linear(1024, 4096)
        # self.layer8 = self._make_layer(block, 1024, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.input_dim_1 = 128
        self.mamba3_1 = nn.Sequential(nn.LayerNorm(self.input_dim_1),
                                      Mamba(d_model=self.input_dim_1, d_state=4, d_conv=2,
                                            expand=1))
        self.input_dim_2 = 512
        self.mamba3_2 = nn.Sequential(nn.LayerNorm(self.input_dim_2),
                                      Mamba(d_model=self.input_dim_2, d_state=4, d_conv=2,
                                            expand=1))
        self.dropout1 = torch.nn.Dropout(0.5)

        # 添加一个多头注意力模块
        # self.attention = nn.MultiheadAttention(embed_dim=1024, num_heads=8)
        # self.attention_layer = AttentionLayer(embed_dim=1024, num_heads=8, feedforward_dim=512)
        # ####self.fft_layer = FourierTransformLayer(embed_dim=1024, feedforward_dim=512)
        # self.fft_layer = FNetEncoderLayer(d_model=1024, dim_feedforward=256)
        self.attention_layer1 = AttentionLayer(embed_dim=128, num_heads=8, feedforward_dim=256)
        self.attention_layer2 = AttentionLayer(embed_dim=512, num_heads=8, feedforward_dim=256)
        # self.attention_layer3 = AttentionLayer(embed_dim=512, num_heads=8, feedforward_dim=256)
        # self.attention_layer4 = AttentionLayer(embed_dim=128, num_heads=8, feedforward_dim=128)
        # self.norm1 = nn.LayerNorm(313)
        # self.norm2 = nn.LayerNorm(79)

        self.fc = nn.Linear(1024 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, 1))
        return nn.Sequential(*layers)

    # out first: torch.Size([32, 64, 2500])
    # out layer1: torch.Size([32, 64, 2500])
    # out layer2: torch.Size([32, 128, 1250])
    # out layer3: torch.Size([32, 256, 625])
    # out layer4: torch.Size([32, 512, 313])
    # out avgpool: torch.Size([32, 512, 1])
    # out shape: torch.Size([32, 512])

    def forward(self, x, include_fc=True):
        out = F.relu(self.bn1(self.conv1(x)))
        # print("out first:", out.shape) 16
        out = self.layer1(out)

        # print("out layer1:", out.shape)
        out = self.layer2(out)
        # [32, 32, 1250]
        # out = self.attention_layer1(out.permute(2, 0, 1)).permute(1, 2, 0)
        # print("out layer2:", out.shape)
        out = self.layer3(out)
        # res_out=out

        # print("out layer3:", out.shape)
        # out = self.attention_layer1(out.permute(2, 0, 1)).permute(1, 2, 0)
        # print("out attention_layer1:", out.shape)
        # print("out layer3:", out.shape)
        # out = out+res_out

        out = self.layer4(out)


        # out = self.attention_layer1(out.permute(2, 0, 1)).permute(1, 2, 0)
        # out = self.mamba3_1(out.permute(2, 0, 1)).permute(1, 2, 0)
        out = self.mamba3_1(out.permute(0,2, 1)).permute(0, 2, 1)
        # out = self.dropout1(out)


        # out = self.attention_layer1(out.permute(2, 0, 1)).permute(1, 2, 0)+out
        # out = self.norm1(self.attention_layer1(out.permute(2, 0, 1)).permute(1, 2, 0)+out)
        out = self.layer5(out)
        # out = self.attention_layer2(out.permute(2, 0, 1)).permute(1, 2, 0)
        out = self.layer6(out)


        # out = self.attention_layer2(out.permute(2, 0, 1)).permute(1, 2, 0)
        # out = self.mamba3_2(out.permute(2, 0, 1)).permute(1, 2, 0)
        out = self.mamba3_2(out.permute(0 ,2, 1)).permute(0, 2, 1)


        # # out = self.attention_layer2(out.permute(2, 0, 1)).permute(1, 2, 0)+out
        # # out = self.norm2(self.attention_layer2(out.permute(2, 0, 1)).permute(1, 2, 0)+out)

        out = self.layer7(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)  # 展平除了批次维度之外的所有维度
        # print("------After flattening:", x.shape)  # 查看展平后的形状

        out = self.input_linear(out).view(out.size(0), 64, 64)
        # # # # # # # 在全连接层之前应用注意力模块
        # out = out.permute(2, 0, 1)  # 调整维度以符合多头注意力的输入要求
        # # # # out, _ = self.attention(out, out, out)
        # out = self.fft_layer(out)
        # # out = self.fft_layer(out)
        # # # # # out = self.attention_layer1(out)
        # # # # # out = self.attention_layer2(out)
        # # # out = self.attention_layer(out)
        # # # # out = self.attention_layer(out)
        # # # # # out = self.attention_layer(out)
        # out = out.permute(1, 2, 0)  # 恢复原始维度

        # if transformer  不要
        # out = self.avgpool(out)
        # # print("out avgpool:", out.shape)
        # out = out.view(out.size(0), -1)
        # # print("out shape:", out.shape)
        # out = self.fc(out) if include_fc else out
        return out


# out first: torch.Size([32, 64, 2500])
# out layer1: torch.Size([32, 64, 2500])
# out layer2: torch.Size([32, 128, 1250])
# out layer3: torch.Size([32, 256, 625])
# out layer4: torch.Size([32, 512, 313])
# out avgpool: torch.Size([32, 512, 1])
# out shape: torch.Size([32, 512, 1])
# x avgpool: torch.Size([32, 36, 128])
# x positional_encoding: torch.Size([32, 36, 128])
# x transformer: torch.Size([32, 36, 128])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, feedforward_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, feedforward_dim, dropout)
            for _ in range(num_layers)
        ])
        self.pos_encoder = PositionalEncoding(embed_dim)

    def forward(self, x):
        x = self.pos_encoder(x)  # 位置编码已在 PositionalEncoding 类中与 x 相加
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = AttentionLayer(embed_dim, num_heads, feedforward_dim, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.attention(x)
        return self.norm(x + self.dropout(x))


if __name__ == '__main__':
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    a = torch.randn((128, 1, 5000)).to("cuda")
    model = ECGFeatureExtractor_mamba(input_channels=1, cnn_output_channels=64, model_dim=128, num_heads=2, num_layers=2).to("cuda")
    # model = ResNet1D(ResNet1DBlock, [2, 2, 2, 2, 2, 2, 2], num_classes=7).to("cuda")

    output = model(a)





