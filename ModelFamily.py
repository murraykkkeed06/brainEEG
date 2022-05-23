
from pickle import TRUE
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from function.modules import Ensure4d, Expression
from function.functions import squeeze_final_output
from torch import nn
from torch.nn import init
from torch import Tensor
from torch.nn.utils import weight_norm
import torch.nn as nn
import pandas as pd
import numpy as np
import torch


class SCCNet(nn.Module):

    def __init__(self, num_classes=4):
        super(SCCNet, self).__init__()
        #input shape B,1,22,438
        self.__name__ = "SCCNet"
        self.spatial = nn.Sequential(
            nn.Conv2d(
                1, 22, kernel_size=(22, 1), padding=(0,0)
            ),
            nn.BatchNorm2d(22),
            nn.Identity()
        )
        self.spatial_temporal = nn.Sequential(
            nn.Conv2d(
                1, 20, kernel_size=(22,12), padding=(0,5)
            ),
            nn.BatchNorm2d(20)
        )
        self.drop_out = nn.Dropout(0.5)
        self.temporal_smoothing = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 62), stride=(1,12)),
        )
        self.classify = nn.Sequential(
            nn.Linear(in_features=20 * 32, out_features=num_classes)
        )

    
        
        
    def forward(self, x):
        out = self.spatial(x)
        out = torch.permute(out,(0,2,1,3))
        out = self.spatial_temporal(out)
        out = torch.square(out)
        out = self.drop_out(out)
        out = torch.permute(out,(0,2,1,3))
        out = self.temporal_smoothing(out)
        out = torch.log10(out)
        out = out.flatten(start_dim=1)
        out = self.classify(out)

        return out


class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()

        self.F1 = 8
        self.F2 = 16
        self.D = 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(self.F1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.F1, self.D*self.F1, (22, 1), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.D*self.F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        )

        self.Conv3 = nn.Sequential(
            nn.Conv2d(self.D*self.F1, self.D*self.F1, (1, 16), padding=(0, 8), groups=self.D*self.F1, bias=False),
            nn.Conv2d(self.D*self.F1, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5)
        )

        self.classifier = nn.Linear(16*13, 4, bias=True)
        #self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.Conv3(x)
        x = x.view(-1, 16*13)
        x = self.classifier(x)
        #x = self.softmax(x)
        return x

class EEGNet_TCN(nn.Module):
    def __init__(self):
        super(EEGNet_TCN, self).__init__()
        self.__name__ = "EEGNet_TCN"
        self.F1 = 24
        self.F2 = 16
        self.D = 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, 32), padding='same', bias=False),
            nn.BatchNorm2d(self.F1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.F1, self.D*self.F1, (22, 1), padding='valid', groups=self.F1, bias=False),
            nn.BatchNorm2d(self.D*self.F1),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.3)
        )

        self.Conv3 = nn.Sequential(
            nn.Conv2d(self.D*self.F1, self.D*self.F1, (1, 16), padding='same', groups=self.D*self.F1, bias=False),
            nn.Conv2d(self.D*self.F1, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.3)
        )

        self.Tcn = TCN()

        self.classifier = nn.Linear(748, 4, bias=True)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.Conv3(x)

        b, c, _, t = x.size()
        eegnet_output = torch.reshape(x,(b,c,t))
        tcn_output = self.Tcn(x)
        concat_output = torch.concat((eegnet_output,tcn_output),dim=1)
        concat_flatten = concat_output.view(-1, 28*17)
        eeg_flatten = eegnet_output.view(-1, 16*17)
        x = torch.concat((concat_flatten,eeg_flatten),dim=1)
        x = self.classifier(x)
        return x


class TCN(nn.Module):
    """Temporal Convolutional Network (TCN) from Bai et al 2018.

    See [Bai2018]_ for details.

    Code adapted from https://github.com/locuslab/TCN/blob/master/TCN/tcn.py

    Parameters
    ---------- 
    n_in_chans: int
        number of input EEG channels
    n_outputs: int
        number of outputs of the decoding task (for example number of classes in
        classification)
    n_filters: int
        number of output filters of each convolution
    n_blocks: int
        number of temporal blocks in the network
    kernel_size: int
        kernel size of the convolutions
    drop_prob: float
        dropout probability
    add_log_softmax: bool
        whether to add a log softmax layer

    References
    ----------
    .. [Bai2018] Bai, S., Kolter, J. Z., & Koltun, V. (2018).
       An empirical evaluation of generic convolutional and recurrent networks
       for sequence modeling.
       arXiv preprint arXiv:1803.01271.
    """
    def __init__(self, n_in_chans=16, n_outputs=4, n_blocks=2
                 , n_filters=12, kernel_size=4,
                 drop_prob=0.3, add_log_softmax=False):
        super().__init__()
   
        t_blocks = nn.Sequential()
        for i in range(n_blocks):
            n_inputs = n_in_chans if i == 0 else n_filters
            dilation_size = 2 ** i
            t_blocks.add_module("temporal_block_{:d}".format(i), TemporalBlock(
                n_inputs=n_inputs,
                n_outputs=n_filters,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation_size,
                padding=(kernel_size - 1) * dilation_size,
                drop_prob=drop_prob
            ))
        self.temporal_blocks = t_blocks
       

    def forward(self, x):
      
        b, c, _, t = x.size()
        x = torch.reshape(x,(b,c,t,1))
        x = x.squeeze(3)
        x = self.temporal_blocks(x)

        return x


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation,
                 padding, drop_prob):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ELU()
        self.dropout1 = nn.Dropout2d(drop_prob)

        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ELU()
        self.dropout2 = nn.Dropout2d(drop_prob)

        self.downsample = (nn.Conv1d(n_inputs, n_outputs, 1)
                           if n_inputs != n_outputs else None)
        self.relu = nn.ELU()

        init.normal_(self.conv1.weight, 0, 0.01)
        init.normal_(self.conv2.weight, 0, 0.01)
        if self.downsample is not None:
            init.normal_(self.downsample.weight, 0, 0.01)

    def forward(self, x):
        #b 64 64, b 20 64 
        out = self.conv1(x)
        #b 20 66, b 20 68
        out = self.chomp1(out)
        #b 20 64, b 20 64
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        #b 20 66, b 20 68 
        out = self.chomp2(out)
        #b 20 64, b 20 64
        out = self.relu2(out)
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)# b 20 64
        return self.relu(out + res)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def extra_repr(self):
        return 'chomp_size={}'.format(self.chomp_size)

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class ShallowConvNet(nn.Module):
    def __init__(self):
        super(ShallowConvNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 40, (1, 13), bias=False)
        self.conv2 = nn.Conv2d(40, 40, (22, 1), bias=False)
        self.Bn1   = nn.BatchNorm2d(40)
        # self.SquareLayer = square_layer()
        self.AvgPool1 = nn.AvgPool2d((1, 35), stride=(1, 7))
        # self.LogLayer = Log_layer()
        self.Drop1 = nn.Dropout(0.25)
        self.classifier = nn.Linear(40*74, 4, bias=True)
        #self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.Bn1(x)
        x = x ** 2
        x = self.AvgPool1(x)
        x = torch.log(x)
        x = self.Drop1(x)
        x = x.view(-1, 40*74)
        x = self.classifier(x)

        #x = self.softmax(x)
        return x

class HSCNN(nn.Module):
    def __init__(self, num_classes=4, BN = TRUE):
        super(HSCNN, self).__init__()
        #input shape B,1,22,438
        self.__name__ = "HSCNN"
        kernel_size = 22
        self.temporal_4_85 = nn.Sequential(
            nn.Conv2d( 1, 10, kernel_size=(1, 85), stride=(1,3)),
            nn.BatchNorm2d(10) if BN else nn.Identity(),
            nn.ELU()
        )
        self.temporal_4_65 = nn.Sequential(
            nn.Conv2d( 1, 10, kernel_size=(1, 65), stride=(1,3)),
            nn.BatchNorm2d(10) if BN else nn.Identity(),
            nn.ELU()
        )
        self.temporal_4_45 = nn.Sequential(
            nn.Conv2d( 1, 10, kernel_size=(1, 45), stride=(1,3)),
            nn.BatchNorm2d(10) if BN else nn.Identity(),
            nn.ELU()
        )
        self.temporal_8_85 = nn.Sequential(
            nn.Conv2d( 1, 10, kernel_size=(1, 85), stride=(1,3)),
            nn.BatchNorm2d(10) if BN else nn.Identity(),
            nn.ELU()
        )
        self.temporal_8_65 = nn.Sequential(
            nn.Conv2d( 1, 10, kernel_size=(1, 65), stride=(1,3)),
            nn.BatchNorm2d(10) if BN else nn.Identity(),
            nn.ELU()
        )
        self.temporal_8_45 = nn.Sequential(
            nn.Conv2d( 1, 10, kernel_size=(1, 45), stride=(1,3)),
            nn.BatchNorm2d(10) if BN else nn.Identity(),
            nn.ELU()
        )
        self.temporal_13_85 = nn.Sequential(
            nn.Conv2d( 1, 10, kernel_size=(1, 85), stride=(1,3)),
            nn.BatchNorm2d(10) if BN else nn.Identity(),
            nn.ELU()
        )
        self.temporal_13_65 = nn.Sequential(
            nn.Conv2d( 1, 10, kernel_size=(1, 65), stride=(1,3)),
            nn.BatchNorm2d(10) if BN else nn.Identity(),
            nn.ELU()
        )
        self.temporal_13_45 = nn.Sequential(
            nn.Conv2d( 1, 10, kernel_size=(1, 45), stride=(1,3)),
            nn.BatchNorm2d(10) if BN else nn.Identity(),
            nn.ELU()
        )


        self.spatial_4_1 = nn.Sequential(
            nn.Conv2d( 10, 10, kernel_size=(kernel_size, 1)),
            nn.BatchNorm2d(10) if BN else nn.Identity(),
            nn.ELU()
        )
        self.spatial_4_2 = nn.Sequential(
            nn.Conv2d( 10, 10, kernel_size=(kernel_size, 1)),
            nn.BatchNorm2d(10) if BN else nn.Identity(),
            nn.ELU()
        )
        self.spatial_4_3 = nn.Sequential(
            nn.Conv2d( 10, 10, kernel_size=(kernel_size, 1)),
            nn.BatchNorm2d(10) if BN else nn.Identity(),
            nn.ELU()
        )
        self.spatial_8_1 = nn.Sequential(
            nn.Conv2d( 10, 10, kernel_size=(kernel_size, 1)),
            nn.BatchNorm2d(10) if BN else nn.Identity(),
            nn.ELU()
        )
        self.spatial_8_2 = nn.Sequential(
            nn.Conv2d( 10, 10, kernel_size=(kernel_size, 1)),
            nn.BatchNorm2d(10) if BN else nn.Identity(),
            nn.ELU()
        )
        self.spatial_8_3 = nn.Sequential(
            nn.Conv2d( 10, 10, kernel_size=(kernel_size, 1)),
            nn.BatchNorm2d(10) if BN else nn.Identity(),
            nn.ELU()
        )
        self.spatial_13_1 = nn.Sequential(
            nn.Conv2d( 10, 10, kernel_size=(kernel_size, 1)),
            nn.BatchNorm2d(10) if BN else nn.Identity(),
            nn.ELU()
        )
        self.spatial_13_2 = nn.Sequential(
            nn.Conv2d( 10, 10, kernel_size=(kernel_size, 1)),
            nn.BatchNorm2d(10) if BN else nn.Identity(),
            nn.ELU()
        )
        self.spatial_13_3 = nn.Sequential(
            nn.Conv2d( 10, 10, kernel_size=(kernel_size, 1)),
            nn.BatchNorm2d(10) if BN else nn.Identity(),
            nn.ELU()
        )

        self.max_pooling = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 6), stride=(1,6)),
            #nn.ELU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=4050, out_features=100),
            
        )

        self.fc2 = nn.Sequential(
            nn.ELU(),
            nn.Linear(in_features=100, out_features=num_classes)
        )

        self.drop_out = nn.Dropout(0.8)

         
        
    def forward(self, x):

        data_4 =x[:,0]
        data_8 = x[:,1]
        data_13 = x[:,2]

        # temporal
        out_4_1 = self.temporal_4_85(data_4)
        out_4_2 = self.temporal_4_65(data_4)
        out_4_3 = self.temporal_4_45(data_4)

        out_8_1 = self.temporal_8_85(data_8)
        out_8_2 = self.temporal_8_65(data_8)
        out_8_3 = self.temporal_8_45(data_8)

        out_13_1 = self.temporal_13_85(data_13)
        out_13_2 = self.temporal_13_65(data_13)
        out_13_3 = self.temporal_13_45(data_13)

        # spatial
        out_4_1 = self.spatial_4_1(out_4_1)
        out_4_2 = self.spatial_4_2(out_4_2)
        out_4_3 = self.spatial_4_3(out_4_3)

        out_8_1 = self.spatial_8_1(out_8_1)
        out_8_2 = self.spatial_8_2(out_8_2)
        out_8_3 = self.spatial_8_3(out_8_3)

        out_13_1 = self.spatial_13_1(out_13_1)
        out_13_2 = self.spatial_13_2(out_13_2)
        out_13_3 = self.spatial_13_3(out_13_3)

        # permute
        out_4_1 = torch.permute(out_4_1,(0,2,1,3))
        out_4_2 = torch.permute(out_4_2,(0,2,1,3))
        out_4_3 = torch.permute(out_4_3,(0,2,1,3))

        out_8_1 = torch.permute(out_8_1,(0,2,1,3))
        out_8_2 = torch.permute(out_8_2,(0,2,1,3))
        out_8_3 = torch.permute(out_8_3,(0,2,1,3))

        out_13_1 = torch.permute(out_13_1,(0,2,1,3))
        out_13_2 = torch.permute(out_13_2,(0,2,1,3))
        out_13_3 = torch.permute(out_13_3,(0,2,1,3))

        # Max pooling
        out_4_1 = self.max_pooling(out_4_1)
        out_4_2 = self.max_pooling(out_4_2)
        out_4_3 = self.max_pooling(out_4_3)

        out_8_1 = self.max_pooling(out_8_1)
        out_8_2 = self.max_pooling(out_8_2)
        out_8_3 = self.max_pooling(out_8_3)

        out_13_1 = self.max_pooling(out_13_1)
        out_13_2 = self.max_pooling(out_13_2)
        out_13_3 = self.max_pooling(out_13_3)

        # flatten
        out_4_1 = out_4_1.flatten(start_dim=1)
        out_4_2 = out_4_2.flatten(start_dim=1)
        out_4_3 = out_4_3.flatten(start_dim=1)

        out_8_1 = out_8_1.flatten(start_dim=1)
        out_8_2 = out_8_2.flatten(start_dim=1)
        out_8_3 = out_8_3.flatten(start_dim=1)

        out_13_1 = out_13_1.flatten(start_dim=1)
        out_13_2 = out_13_2.flatten(start_dim=1)
        out_13_3 = out_13_3.flatten(start_dim=1)
        
        # fc
        out = torch.cat((out_4_1, out_4_2, out_4_3, out_8_1, out_8_2, out_8_3, out_13_1, out_13_2, out_13_3), axis=1)
        out = self.fc1(out)
        out = self.drop_out(out)
        out = self.fc2(out)


        return out


class TSception(nn.Module):
    def conv_block(self, in_chan, out_chan, kernel, step, pool):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=kernel, stride=step, padding=0),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))

    def __init__(self, num_classes, input_size, sampling_rate, num_T, num_S, hidden, dropout_rate):
        # input_size: 1 x EEG channel x datapoint
        super(TSception, self).__init__()
        self.inception_window = [0.5, 0.25, 0.125]
        self.pool = 8
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = self.conv_block(1, num_T, (1, int(self.inception_window[0] * sampling_rate)), 1, self.pool)
        self.Tception2 = self.conv_block(1, num_T, (1, int(self.inception_window[1] * sampling_rate)), 1, self.pool)
        self.Tception3 = self.conv_block(1, num_T, (1, int(self.inception_window[2] * sampling_rate)), 1, self.pool)

        self.Sception1 = self.conv_block(num_T, num_S, (int(input_size[1]), 1), 1, int(self.pool*0.25))
        self.Sception2 = self.conv_block(num_T, num_S, (int(input_size[1] * 0.5), 1), (int(input_size[1] * 0.5), 1),
                                         int(self.pool*0.25))
        self.fusion_layer = self.conv_block(num_S, num_S, (3, 1), 1, 2)
        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_S)
        self.BN_fusion = nn.BatchNorm2d(num_S)

        size = self.get_size(input_size)
        self.fc = nn.Sequential(
            nn.Linear(size[1], hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_ = z
        z = self.Sception2(out)
        out_ = torch.cat((out_, z), dim=2)

        out = self.BN_s(out_)
        out = self.fusion_layer(out)
        out = self.BN_fusion(out)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out

    def get_size(self, input_size):
        # here we use an array with the shape being
        # (1(mini-batch),1(convolutional channel),EEG channel,time data point)
        # to simulate the input data and get the output size
        data = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
        y = self.Tception1(data)
        out = y
        y = self.Tception2(data)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(data)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_final = z
        z = self.Sception2(out)
        out_final = torch.cat((out_final, z), dim=2)
        out = self.BN_s(out_final)
        out = self.fusion_layer(out)
        out = self.BN_fusion(out)
        out = out.view(out.size()[0], -1)
        return out.size()

class Square(nn.Module):
    def forward(self, x):
        return torch.square(x)


class Log(nn.Module):
    def forward(self, x):
        return torch.log(x)

class SCCTransformer(nn.Module):

    def __init__(self, num_classes=4):
        super(SCCTransformer, self).__init__()
        #input shape B,1,22,438
        self.__name__ = "SCCTransformer"

        self.SCCNet = nn.Sequential(
            nn.Conv2d(1, 22, (22, 1)),
            nn.BatchNorm2d(22),
            nn.Conv2d(22, 20, (1, 12), padding=(0, 5)),
            nn.BatchNorm2d(20),
            Square(),
            nn.Dropout(0.5),
            nn.AvgPool2d((1, 62), stride=(1, 12)),
            Log(),
            nn.Flatten(),
            nn.Linear(120, 64, bias=True)
        )

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8)

        self.classifier = nn.Sequential(
            nn.TransformerEncoder(self.encoder_layer, num_layers=2),
            nn.Flatten(),
            nn.Linear(704, 64, bias=True),
            nn.Dropout(0.8),
            nn.ReLU(),
            nn.Linear(64, 4, bias=True)
        )
        
        0
        self.d1 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(11, num_classes)

    def forward(self, x):

        out_seq = []
        for i in range(11):
            out_seq.append(self.SCCNet(x[:,:,:,31*i:31*i+128]))
        out = torch.stack(out_seq, 1)

        out = self.classifier(out)

        return out
