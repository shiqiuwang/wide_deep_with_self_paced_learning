import torch.nn as nn
from torch.nn.utils import weight_norm
import torch
import torch.nn.functional as F
import math
import numpy as np


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        """

        :param chomp_size: 填充值 1 int
        """
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        :param n_inputs: 输入通道数 int
        :param n_outputs: 输出通道数 int
        :param kernel_size: 卷积核大小 2 int
        :param stride: 步幅 1 int
        :param dilation: 扩张大小 list
        :param padding: 填充值 1 int
        :param dropout: dropout比率 0.2 float
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, max_length=39, attention=False):
        """

        :param num_inputs: 输入的特征的个数 int
        :param num_channels: 网络层的通道数
        :param kernel_size: 卷积核的大小 2 int
        :param dropout: dropout比率 0.2
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)  # 网络层的数量 4 int
        if attention == True:
            layers += [AttentionBlock(max_length, max_length, max_length)]
        for i in range(num_levels):
            dilation_size = 2 ** i  # 扩展大小，每一层都为2的次方
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  
            out_channels = num_channels[i]  # 输出通道数 int
            # if attention == True:
            #     layers += [AttentionBlock(max_length, max_length, max_length)]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
            # if attention == True:
            #     layers += [AttentionBlock(max_length, max_length, max_length)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class AttentionBlock(nn.Module):
    """An attention mechanism similar to Vaswani et al (2017)
    The input of the AttentionBlock is `BxTxD` where `B` is the input
    minibatch size, `T` is the length of the sequence `D` is the dimensions of
    each feature.
    The output of the AttentionBlock is `BxTx(D+V)` where `V` is the size of the
    attention values.
    Arguments:
        dims (int): the number of dimensions (or channels) of each element in
            the input sequence
        k_size (int): the size of the attention keys
        v_size (int): the size of the attention values
        seq_len (int): the length of the input and output sequences
    """

    def __init__(self, dims, k_size, v_size, seq_len=None):
        super(AttentionBlock, self).__init__()
        self.key_layer = nn.Linear(dims, k_size)
        self.query_layer = nn.Linear(dims, k_size)
        self.value_layer = nn.Linear(dims, v_size)
        self.sqrt_k = math.sqrt(k_size)

    def forward(self, minibatch):
        keys = self.key_layer(minibatch)
        queries = self.query_layer(minibatch)
        values = self.value_layer(minibatch)
        logits = torch.bmm(queries, keys.transpose(2, 1))
        # Use numpy triu because you can't do 3D triu with PyTorch
        # TODO: using float32 here might break for non FloatTensor inputs.
        # Should update this later to use numpy/PyTorch types of the input.
        mask = np.triu(np.ones(logits.size()), k=1).astype('uint8')
        mask = torch.from_numpy(mask).cuda()
        # do masked_fill_ on data rather than Variable because PyTorch doesn't
        # support masked_fill_ w/-inf directly on Variables for some reason.
        logits.data.masked_fill_(mask, float('-inf'))
        probs = F.softmax(logits, dim=1) / self.sqrt_k
        read = torch.bmm(probs, values)
        return minibatch + read
