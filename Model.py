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
