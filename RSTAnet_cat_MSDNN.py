
import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba, Block
import torch.nn.functional as F

k1, p1 = 3, 1
k2, p2 = 5, 2
k3, p3 = 9, 4
k4, p4 = 17, 8

class SELayer1D(nn.Module):

    def __init__(self, nChannels, reduction=16):
        super(SELayer1D, self).__init__()
        self.globalavgpool = nn.AdaptiveAvgPool1d(1)
        self.se_block = nn.Sequential(
            nn.Linear(nChannels, nChannels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nChannels // reduction, nChannels, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        alpha = torch.squeeze(self.globalavgpool(x))
        alpha = self.se_block(alpha)
        alpha = torch.unsqueeze(alpha, -1)
        out = torch.mul(x, alpha)
        return out

class BranchConv1D(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(BranchConv1D, self).__init__()
        C = out_channels // 3
        # self.b1 = nn.Conv1d(in_channels, C, k1, stride, p1, bias=False)#p1：填充（padding）的大小，如果设置为 0 表示不填充。
        self.b2 = nn.Conv1d(in_channels, C, k2, stride, p2, bias=False)
        self.b3 = nn.Conv1d(in_channels, C, k3, stride, p3, bias=False)
        self.b4 = nn.Conv1d(in_channels, C, k4, stride, p4, bias=False)

    def forward(self, x):
        out = torch.cat([self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return out

class BasicBlock1D(nn.Module):

    def __init__(self, in_channels, out_channels, drop_out_rate, stride):
        super(BasicBlock1D, self).__init__()
        self.operation = nn.Sequential(
                BranchConv1D(in_channels, out_channels, stride),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(drop_out_rate),
                BranchConv1D(out_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels),
                SELayer1D(out_channels))

        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut.add_module('MaxPool', nn.MaxPool1d(stride, ceil_mode=True))
        if in_channels != out_channels:
            self.shortcut.add_module('ShutConv', nn.Conv1d(in_channels, out_channels, 1))
            self.shortcut.add_module('ShutBN', nn.BatchNorm1d(out_channels))

    def forward(self, x):
        operation = self.operation(x)
        shortcut = self.shortcut(x)
        out = torch.relu(operation + shortcut)
        return out


class MSDNN(nn.Module):

    def __init__(self, num_classes=1, init_channels=1, growth_rate=12, base_channels=48,
                 stride=2, drop_out_rate=0.2):
        super(MSDNN, self).__init__()
        self.num_channels = init_channels
        block_n = 6
        block_c = [base_channels + i * growth_rate for i in range(block_n)]

        self.blocks = nn.Sequential()
        for i, C in enumerate(block_c):
            module = BasicBlock1D(self.num_channels, C, drop_out_rate, stride)
            self.blocks.add_module("block{}".format(i), module)
            self.num_channels = C

        # module = nn.AdaptiveAvgPool1d(1)
        # self.blocks.add_module("GlobalAvgPool", module)

        # self.fc = nn.Sequential(nn.Linear(self.num_channels, num_classes),
        #                          nn.Sigmoid())

    def forward(self, x):
        out = self.blocks(x)
        out = torch.squeeze(out)
        # out = self.fc(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNet1D, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(16)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.input_dim_1 = 108
        self.mamba3_1 = nn.Sequential(nn.LayerNorm(self.input_dim_1),
                                      Mamba(d_model=self.input_dim_1, d_state=4, d_conv=2,
                                            expand=1))


        self.input_dim_2 = 108
        self.mamba3_2 = nn.Sequential(nn.LayerNorm(self.input_dim_2),
                                      Mamba(d_model=self.input_dim_2, d_state=4, d_conv=2,
                                            expand=1))

        self.dropout1 = torch.nn.Dropout(0.5)
        self.multi_scale = MSDNN()
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

        self.fc = nn.Linear(self.input_dim_1, num_classes)

    # out first: torch.Size([32, 64, 2500])
    # out layer1: torch.Size([32, 64, 2500])
    # out layer2: torch.Size([32, 128, 1250])
    # out layer3: torch.Size([32, 256, 625])
    # out layer4: torch.Size([32, 512, 313])
    # out avgpool: torch.Size([32, 512, 1])
    # out shape: torch.Size([32, 512])

    def forward(self, x, include_fc=True):
        out = self.multi_scale(x)

        out_31 = self.mamba3_1(out.permute(0, 2, 1)).permute(0, 2, 1)

        out_reverse = torch.flip(out, dims=[2])
        out_32 = self.mamba3_2(out_reverse.permute(0, 2, 1)).permute(0, 2, 1)
        out = out_31 + out_32
        # print("out mamba3_2:", out.shape)

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
    # model = ResNet1D(ResNet1DBlock, [2, 2, 2, 2,2,2,2], num_classes=7).to("cuda")
    model = ResNet1D(num_classes=7).to("cuda")
    output = model(x, include_fc=True)
    print("Output shape:", output.shape)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total / 1e6))
    print('abc')
    new = getModelSize(model)
    compute_flops(model, (1, 1, 5000))
    summary(model,x)
    flops, params = profile(model, inputs=(x,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    # 生成计算图
    # graph = make_dot(output, params=dict(list(model.named_parameters()) + [('input', x)]))
    #
    # # 保存图像
    # graph.render('model_visualization', format='png')
