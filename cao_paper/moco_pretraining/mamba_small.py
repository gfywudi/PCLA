import torch
from mamba_ssm import Mamba
# from RevIN.RevIN import RevIN
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba, Block
from ECG.model.PTB_XL_resnet import resnet_mu
from ECG.model.MSDNN_use_to_cat import MSDNN

import sys
sys.path.append("/home/guofengyi/code/mamba-main")
from mamba_ssm.ops.triton.layernorm import RMSNorm

class Model(torch.nn.Module):
    def __init__(self, num_classes=15):
        super(Model, self).__init__()

        # if self.configs.revin == 1:
        #     self.revin_layer = RevIN(self.configs.enc_in)
        self.ch_ind = 1
        self.residual = 1
        self.layer_num = 4
        self.conv_out = 20 #625resnet
        self.input_dim = 256
        self.lead_number = 176 #48resnet
        # self.conv1 = nn.Sequential(
        #     nn.Conv1d(in_channels=12, out_channels=12, kernel_size=10, padding=0,groups=12),
        #     nn.BatchNorm1d(num_features=12, eps=1e-05, momentum=0.1, affine=True),
        #     nn.LeakyReLU())
        self.batchnorm1 = nn.BatchNorm1d(num_features=12, eps=1e-05, momentum=0.1, affine=True)
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
        self.mamba3_4 = nn.Sequential(nn.LayerNorm(self.input_dim),
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
        self.mamba4_4 = nn.Sequential(nn.LayerNorm(self.lead_number),
                                      Mamba(d_model=self.lead_number, d_state=4, d_conv=2,
                                            expand=1))

        self.fc3 = nn.Sequential(nn.Linear(16*self.lead_number, num_classes),
                                 nn.Sigmoid())
        self.fc = nn.Linear(16*self.lead_number, num_classes)
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
        self.resnet = resnet_mu()
        self.multi_scale = MSDNN()
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
        x_res1 = x
        x3 = self.mamba3(x)
        x3 = self.mamba3_1(x3)
        x3 = self.mamba3_2(x3)
        x3 = self.mamba3_3(x3)
        # x3 = self.mamba3_4(x3)
        x3 = x3+x_res1



        x3 = self.dropout2(x3)

        if self.ch_ind == 1:
            x4 = torch.permute(x, (0, 2, 1))
        else:
            x4 = x
        x_res2 = x4
        x4 = self.mamba4(x4)
        x4 = self.mamba4_1(x4)
        x4 = self.mamba4_2(x4)
        x4 = self.mamba4_3(x4)
        # x4 = self.mamba4_4(x4)
        x4 = x4 + x_res2



        x4 = self.dropout3(x4)
        if self.ch_ind == 1:
            x4 = torch.permute(x4, (0, 2, 1))

        # x4 = torch.cat([x3, x4], dim=2)
        x4 = x3+x4


        x = self.lin5(x4)
        out = x.view(x.shape[0], -1)
        # out = self.fc3(out)
        out = self.fc(out)



        return out



if __name__=='__main__':
    a = torch.randn((128,12,5000)).to("cuda")
    model = Model().to("cuda")
    output = model(a)