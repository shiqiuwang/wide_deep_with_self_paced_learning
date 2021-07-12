# from torch import nn
# from tcn import TemporalConvNet
# from torch.nn.utils import weight_norm
#
#
# class TCN(nn.Module):
#     def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
#         super(TCN, self).__init__()
#         self.conv = weight_norm(nn.Conv1d(in_channels=1, out_channels=1, kernel_size=(100,), stride=100))
#         self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
#         self.linear = nn.Linear(num_channels[-1], output_size)
#
#     def init_weights(self):
#         self.conv.weight.data.normal_(0, 0.01)
#
#     def forward(self, inputs):
#         """Inputs have to have dimension (N, C_in, L_in)"""
#         out = self.conv(inputs)
#         y1 = self.tcn(out)  # input should have dimension (N, C, L)
#         o = self.linear(y1[:, :, -1])
#         return o

# from torch import nn
# from tcn import TemporalConvNet
# import torch.nn.functional as F
# import torch
# from torch.nn.utils import weight_norm
#
#
# class TCN(nn.Module):
#     def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
#         super(TCN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 64, (3, 100), stride=1)
#         self.conv2 = nn.Conv2d(1, 64, (4, 100), stride=1)
#         self.conv3 = nn.Conv2d(1, 64, (5, 100), stride=1)
#         self.Max1_pool = nn.MaxPool2d((39 - 3 + 1, 1))  # (kernel_size,stride)
#         self.Max2_pool = nn.MaxPool2d((39 - 4 + 1, 1))
#         self.Max3_pool = nn.MaxPool2d((39 - 5 + 1, 1))
#         # self.conv = weight_norm(nn.Conv1d(in_channels=1, out_channels=1, kernel_size=(100,), stride=100))
#         self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
#         self.linear = nn.Linear(num_channels[-1], output_size)
#
#     # def init_weights(self):
#     #     self.conv1.weight.data.normal_(0, 0.01)
#     #     self.conv2.weight.data.normal_(0, 0.01)
#     #     self.conv3.weight.data.normal_(0, 0.01)
#     #     self.Max1_pool.weight.data.normal_(0, 0.01)
#
#     def forward(self, inputs):
#         """Inputs have to have dimension (N, C_in, L_in)"""
#         batch = inputs.shape[0]
#         # Convolution
#         x1 = F.relu(self.conv1(inputs))
#         x2 = F.relu(self.conv2(inputs))
#         x3 = F.relu(self.conv3(inputs))
#
#         # Pooling
#         x1 = self.Max1_pool(x1)
#         x2 = self.Max2_pool(x2)
#         x3 = self.Max3_pool(x3)
#
#         # capture and concatenate the features
#         x = torch.cat((x1, x2, x3), -1)
#         x = x.view(batch, 1, -1)
#         y1 = self.tcn(x)  # input should have dimension (N, C, L)
#         o = self.linear(y1[:, :, -1])
#         return o
from torch import nn
from torch import nn
from torch import nn
from tcn import TemporalConvNet
import torch.nn.functional as F
import torch
from torch.nn.utils import weight_norm


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.fc = nn.Linear(3900, output_size)
        self.init_weights()

    def init_weights(self):
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.normal_(0, 0.01)

    def forward(self, inputs):
        y = self.tcn(inputs.transpose(1, 2))  # input should have dimension (N, C, L)#(16,27,100)
        y = y.contiguous().view(128, -1)

        y = self.fc(y)

        return y
