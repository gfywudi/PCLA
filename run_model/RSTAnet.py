import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchviz import make_dot
import torch.fft as fft
from mamba_ssm.modules.mamba_simple import Mamba, Block


# class FourierTransformLayer(nn.Module):
#     def __init__(self, embed_dim, feedforward_dim, dropout=0.1):
#         super(FourierTransformLayer, self).__init__()
#         self.embed_dim = embed_dim
#         self.feedforward = nn.Sequential(
#             nn.Linear(embed_dim, feedforward_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(feedforward_dim, embed_dim),
#         )
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         # 假设 x 的形状为 (batch_size, channels, sequence_length)
#         # 调整维度以符合傅里叶变换的输入要求
#         x = x.permute(0, 2, 1)  # 变为 (batch_size, sequence_length, channels)

#         # 对序列维度进行傅里叶变换
#         fft_output = torch.fft.fft(x, dim=1)
# print("fft_torch.fft.fft:", fft_output.shape)

#         # 取绝对值（幅度）
#         fft_output = torch.abs(fft_output)

#         # 恢复原始维度
#         fft_output = fft_output.permute(0, 2, 1)  # 变回 (batch_size, channels, sequence_length)
# print("fft_output_torch.abs:", fft_output.shape) # fft_output: torch.Size([32, 1024, 40])

#         x = self.norm1(x + self.dropout(fft_output))
#         print("x:", x.shape)

#         # 前馈网络
#         ff_output = self.feedforward(x)
#         x = self.norm2(x + self.dropout(ff_output))
#         return x


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
        # self.attention_layer1 = AttentionLayer(embed_dim=128, num_heads=8, feedforward_dim=256)
        # self.attention_layer2 = AttentionLayer(embed_dim=512, num_heads=8, feedforward_dim=256)
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
        # print("out first:", out.shape)
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
        # print("out layer4:", out.shape)


        # out = self.attention_layer1(out.permute(2, 0, 1)).permute(1, 2, 0)
        # out = self.mamba3_1(out.permute(2, 0, 1)).permute(1, 2, 0)
        out = self.mamba3_1(out.permute(0,2, 1)).permute(0, 2, 1)
        # out = self.dropout1(out)
        # print("out mamba3_1:", out.shape)


        # out = self.attention_layer1(out.permute(2, 0, 1)).permute(1, 2, 0)+out
        # out = self.norm1(self.attention_layer1(out.permute(2, 0, 1)).permute(1, 2, 0)+out)
        out = self.layer5(out)
        # print("out layer5:", out.shape)
        # out = self.attention_layer2(out.permute(2, 0, 1)).permute(1, 2, 0)
        out = self.layer6(out)
        # print("out layer6:", out.shape)


        # out = self.attention_layer2(out.permute(2, 0, 1)).permute(1, 2, 0)
        # out = self.mamba3_2(out.permute(2, 0, 1)).permute(1, 2, 0)
        out = self.mamba3_2(out.permute(0 ,2, 1)).permute(0, 2, 1)
        # print("out mamba3_2:", out.shape)


        # # out = self.attention_layer2(out.permute(2, 0, 1)).permute(1, 2, 0)+out
        # # out = self.norm2(self.attention_layer2(out.permute(2, 0, 1)).permute(1, 2, 0)+out)

        out = self.layer7(out)
        # out = self.mamba3_2(out.permute(0 ,2, 1)).permute(0, 2, 1)

        # print("out layer7:", out.shape)

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
        out = self.avgpool(out)
        # print("out avgpool:", out.shape)
        out = out.view(out.size(0), -1)
        # print("out shape:", out.shape)
        out = self.fc(out) if include_fc else out
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





if __name__ == "__main__":
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    from torchsummary import summary
    from torchstat import stat
    from thop import profile


    def getModelSize(model):
        param_size = 0
        param_sum = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
            param_sum += param.nelement()
        buffer_size = 0
        buffer_sum = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            buffer_sum += buffer.nelement()
        all_size = (param_size + buffer_size) / 1024 / 1024
        print('模型总大小为：{:.3f}MB'.format(all_size))
        return (param_size, param_sum, buffer_size, buffer_sum, all_size)


    def compute_flops(model, input_size):
        def count_flops(module, input, output):
            # You can extend this function to count FLOPs for different layer types
            if isinstance(module, nn.Linear):
                in_features = module.in_features
                out_features = module.out_features
                flops = 2 * in_features * out_features  # For each element: one multiplication + one addition
                print(f"Linear layer FLOPs: {flops}")

        model.apply(lambda m: m.register_forward_hook(count_flops))
        x = torch.randn(1, 1, 5000).to("cuda")
        with torch.no_grad():
            model(x)
    # resnet = ResNet1D(ResNet1DBlock, [2, 2, 2, 2], num_classes=2)
    # transformer = TransformerEncoder(input_dim=512, num_heads=8, dim_feedforward=2048, num_layers=6)
    # model = ResNet1DTransformer(resnet, transformer, num_classes=2)
    # print(model)

    # 测试模型输入输出
    x = torch.randn(32, 1, 5000).to("cuda")
    model = ResNet1D(ResNet1DBlock, [2, 2, 2, 2,2,2,2], num_classes=7).to("cuda")
    output = model(x, include_fc=True)
    print("Output shape:", output.shape)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total / 1e6))
    print('abc')
    new = getModelSize(model)
    compute_flops(model, (1, 10))
    summary(model,x)
    flops, params = profile(model, inputs=(x,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    # 生成计算图
    # graph = make_dot(output, params=dict(list(model.named_parameters()) + [('input', x)]))
    #
    # # 保存图像
    # graph.render('model_visualization', format='png')
