from torch import nn
import torch
# import torch.nn.functional as F


class FM(nn.Module):
    def __init__(self, n=None, k=None, pre_dim=6):
        """
        :param n: 数据集的特征数，即列数，X.shape[1]
        :param k: 因子分解机中的因子数目
        :param pre_dim:预测结果是一个只有1列的二维张量
        """
        super(FM, self).__init__()
        self.n = n
        self.v = nn.Parameter(torch.randn(n, k), requires_grad=True)
        self.linear = nn.Linear(n, pre_dim)

    def forward(self, X):
        out_1 = torch.matmul(X.view(-1, self.n), self.v.float()).pow(2).sum(dim=1, keepdim=True)  # S1^2
        out_2 = torch.matmul(X.view(-1, self.n).pow(2), self.v.pow(2).float()).sum(dim=1, keepdim=True)  # S2

        out_interaction = 0.5 * (out_1 - out_2).float()
        out_linear = self.linear(X.float()).float()
        out = (out_linear + out_interaction).float()
        # out = F.softmax(out)

        return out

