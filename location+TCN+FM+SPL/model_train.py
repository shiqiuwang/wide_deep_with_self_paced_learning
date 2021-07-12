import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from my_dataset import MyDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from wide_fm import FM
from SPLD import spld
import matplotlib.pyplot as plt
import os

device5 = torch.device("cuda:2")

from deep_tcn import TCN
import numpy as np
import argparse
import random
from generate_url_datasets import url_train_data, url_valid_data

parser = argparse.ArgumentParser(description='spld for url')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 1128)')
parser.add_argument('--dropout', type=float, default=0.05,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--ksize', type=int, default=2,
                    help='kernel size (default: 3)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--wide_lr', type=float, default=0.001,
                    help='initial learning rate (default: 0.5)')
parser.add_argument('--deep_lr', type=float, default=0.001,
                    help='initial learning rate(default:0.1)')
parser.add_argument('--wide_optim', type=str, default='SGD',
                    help='optimizer to use on wide(default:SGD)')
parser.add_argument('--deep_optim', type=str, default='Adam',
                    help='optimizer to use on deep (default: Adam)')
parser.add_argument('--nhid', type=int, default=100,
                    help='number of hidden units per layer (default: 50)')
parser.add_argument('--seed', type=int, default=2020,
                    help='random seed (default: 2020)')
parser.add_argument('--epoch', type=int, default=20,
                    help='the number of training(default:20)')

args = parser.parse_args()

# 设置随机种子
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
# torch.backends.cudnn.deterministic = True

# if torch.cuda.is_available():
#     if not args.cuda:
#         print("WARNING: You have a CUDA device, so you should probably run with --cuda")

batch_size = args.batch_size  # min-batch数量
n_classes = 6  # 分类类别数
input_channels = 100  # 输入通道数
seq_length_at_deep = 3900  # deep部分的向量的长度，即有所少个神经元

# 自步学习超参数
lam = 0.05
gamma = 0.0
# 自步学习参数的学习率调整参数
u1 = 2
u2 = 2

print(args)
# -----------------------------------------------------------第一步：数据------------------------------------------
# 构建MyDataset实例
train_data = MyDataset(url_train_data)
valid_data = MyDataset(url_valid_data)

# 构建DataLoader
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size)

# 每一个隐藏层的神经元的个数列表
channel_sizes = [100] * 4

# 卷积核大小
kernel_size = args.ksize

# 初始化v_star
v_star = torch.randint(low=0, high=2, size=(len(train_loader), batch_size), dtype=torch.float)

# 模型
# ---------------------------------------------------------第二步：模型----------------------------------------------
wide_model = FM(34, 10)
deep_model = TCN(100, 6, channel_sizes, kernel_size=2, dropout=args.dropout)

# v_star2 = np.random.randint(low=0, high=2, size=((len(train_data) % (batch_size)),))
# v_star = []
# for i in range(len(v_star1)):
#     v_star.append(v_star1[i])
# v_star.append(v_star2)

# model2 = TextCnn()

wide_model.to(device5)
deep_model.to(device5)

# 损失函数
# -----------------------------------------------------第三步：损失函数-----------------------------------------------
criterion = nn.CrossEntropyLoss(reduce=False)

# 优化器
# ----------------------------------------------------第四步：优化器-------------------------------------------------
wide_lr = args.wide_lr
wide_optimizer = getattr(optim, args.wide_optim)(wide_model.parameters(), lr=wide_lr)

deep_lr = args.deep_lr
deep_optimizer = getattr(optim, args.deep_optim)(deep_model.parameters(), lr=deep_lr)

wide_scheduler = torch.optim.lr_scheduler.StepLR(wide_optimizer, step_size=2, gamma=0.1)
deep_scheduler = torch.optim.lr_scheduler.StepLR(deep_optimizer, step_size=2, gamma=0.1)

# 训练
# --------------------------------------------------第五步：训练-----------------------------------------------------
train_loss_curve = list()
valid_loss_curve = list()

train_acc_curve = list()
valid_acc_curve = list()

train_acc_curve0 = []
train_acc_curve1 = []
train_acc_curve2 = []
train_acc_curve3 = []
train_acc_curve4 = []
train_acc_curve5 = []

valid_acc_curve0 = []
valid_acc_curve1 = []
valid_acc_curve2 = []
valid_acc_curve3 = []
valid_acc_curve4 = []
valid_acc_curve5 = []

train_real_label = []
train_pre_label = []

valid_real_label = []
valid_pre_label = []

select_idx_list = []
samples_label_arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
record = []
types = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

# ff = open("./results/select_idx.txt", mode='w', encoding='utf-8')

for epoch in range(args.epoch):
    train_loss = 0
    total = 0
    correct = 0

    total0 = 0
    total1 = 0
    total2 = 0
    total3 = 0
    total4 = 0
    total5 = 0

    correct0 = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    correct4 = 0
    correct5 = 0

    wide_model.train()
    deep_model.train()

    for i, data in enumerate(train_loader):
        # samples_label = data[1].numpy()  # 每一个样本对应的类别 ndarray
        # samples_label_arr = np.array(list(set(samples_label)))
        samples_label = data[0][:, -1].numpy().tolist()
        # if i == 0:
        #     record.append(v_star[i])
        # 前向传播

        wide_data, deep_data, target = data[0][:, 0:34].to(device5), data[0][:, 34:-1].to(device5), data[0][:,
                                                                                                    -1].to(
            device5)
        v_star[i] = v_star[i].to(device5)
        deep_data = deep_data.view(128, 39, 100)
        wide_out = wide_model(wide_data)
        deep_out = deep_model(deep_data)

        outputs = wide_out + deep_out

        # 反向传播
        wide_optimizer.zero_grad()
        deep_optimizer.zero_grad()

        loss = criterion(outputs, target.long())

        loss1 = torch.matmul(v_star[i], loss.cpu()) - lam * v_star[i].sum()

        loss2 = torch.tensor(0, dtype=torch.float32)

        for group_id in samples_label_arr:
            idx_for_each_group = np.where(samples_label == group_id)[0]
            loss_for_each_group = torch.tensor(0, dtype=torch.float32)
            for idx in idx_for_each_group:
                loss_for_each_group += (v_star[i][idx] ** 2)
            loss2 += torch.sqrt(loss_for_each_group)
        loss2.to(device5)

        # 计算E
        E = loss1 - gamma * loss2
        E.backward()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(wide_model.parameters(), args.clip)
            torch.nn.utils.clip_grad_norm_(deep_model.parameters(), args.clip)
        wide_optimizer.step()
        deep_optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        for target_val in target.long().cpu().numpy():
            train_real_label.append(target_val)
        for pre_val in predicted.cpu().numpy():
            train_pre_label.append(pre_val)

        total += target.size(0)

        correct += (predicted == target.long()).cpu().squeeze().sum().numpy()

        type_index0 = np.where(target.cpu().numpy() == 0.0)
        total0 += len(type_index0[0])
        for index in type_index0[0]:
            if predicted[index] == target.long()[index]:
                correct0 += 1

        type_index1 = np.where(target.cpu().numpy() == 1.0)
        total1 += len(type_index1[0])
        for index in type_index1[0]:
            if predicted[index] == target.long()[index]:
                correct1 += 1

        type_index2 = np.where(target.cpu().numpy() == 2.0)
        total2 += len(type_index2[0])
        for index in type_index2[0]:
            if predicted[index] == target.long()[index]:
                correct2 += 1

        type_index3 = np.where(target.cpu().numpy() == 3.0)
        total3 += len(type_index3[0])
        for index in type_index3[0]:
            if predicted[index] == target.long()[index]:
                correct3 += 1

        type_index4 = np.where(target.cpu().numpy() == 4.0)
        total4 += len(type_index4[0])
        for index in type_index4[0]:
            if predicted[index] == target.long()[index]:
                correct4 += 1

        type_index5 = np.where(target.cpu().numpy() == 5.0)
        total5 += len(type_index5[0])
        for index in type_index5[0]:
            if predicted[index] == target.long()[index]:
                correct5 += 1

        train_loss += loss.mean().item()
        train_loss_curve.append(loss.mean().item())
        train_acc_curve.append(correct / total)

        new_wide_out_pre = wide_model(wide_data)
        new_deep_out_pre = deep_model(deep_data)
        new_output_pre = new_wide_out_pre + new_deep_out_pre

        new_loss = criterion(new_output_pre, target.long())
        # print("第{}个epoch的第{}个batch对应的loss是{}".format(epoch, i, new_loss.cpu().detach().numpy()))

        selected_idx_arr = spld(new_loss.reshape(new_loss.size()[0], ), samples_label, lam, gamma)
        # print("第{}个epoch的第{}个batch下选中的样本的索引是{}".format(epoch, i, selected_idx_arr))
        select_idx_list.append(selected_idx_arr)

        v_star[i] = torch.zeros((new_loss.size()[0],), dtype=torch.float32)

        for selected_idx in selected_idx_arr:
            v_star[i][selected_idx] = 1

        if (i + 1) % args.log_interval == 0:
            train_loss_mean = train_loss / args.log_interval
            print(
                "Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3} Loss:{:.4f} Acc:{:.2%}".format(epoch, args.epoch,
                                                                                                      i + 1, len(
                        train_loader), train_loss_mean, correct / total))
            train_loss = 0
    if total0 == 0:
        train_acc_curve0.append(0.0)
    else:
        train_acc_curve0.append(correct0 / total0)

    if total1 == 0:
        train_acc_curve1.append(0.0)
    else:
        train_acc_curve1.append(correct1 / total1)

    if total2 == 0:
        train_acc_curve2.append(0.0)
    else:
        train_acc_curve2.append(correct2 / total2)

    if total3 == 0:
        train_acc_curve3.append(0.0)
    else:
        train_acc_curve3.append(correct3 / total3)

    if total4 == 0:
        train_acc_curve4.append(0.0)
    else:
        train_acc_curve4.append(correct4 / total4)

    if total5 == 0:
        train_acc_curve5.append(0.0)
    else:
        train_acc_curve5.append(correct5 / total5)

    wide_scheduler.step()  # 更新wide端学习率
    deep_scheduler.step()  # 更新deep端学习率
    lam = u1 * lam
    gamma = u2 * gamma

    # 验证模型
    if (epoch + 1) % 1 == 0:
        correct_val = 0
        total_val = 0

        correct0_val = 0
        correct1_val = 0
        correct2_val = 0
        correct3_val = 0
        correct4_val = 0
        correct5_val = 0

        total0_val = 0
        total1_val = 0
        total2_val = 0
        total3_val = 0
        total4_val = 0
        total5_val = 0

        loss_val = 0
        wide_model.eval()
        deep_model.eval()

        with torch.no_grad():
            for j, data in enumerate(valid_loader):
                wide_data, deep_data, target = data[0][:, 0:34].to(device5), data[0][:, 34:-1].to(device5), data[0][:,
                                                                                                            -1].to(
                    device5)
                deep_data = deep_data.view(128, 39, 100)
                wide_out = wide_model(wide_data)
                deep_out = deep_model(deep_data)

                outputs = wide_out + deep_out

                loss = criterion(outputs, target.long())

                _, predicted = torch.max(outputs.data, 1)
                for tar_val in target.long().cpu().numpy():
                    valid_real_label.append(tar_val)
                for predict_val in predicted.cpu().numpy():
                    valid_pre_label.append(predict_val)

                total_val += target.size(0)
                correct_val += (predicted == target.long()).cpu().squeeze().sum().numpy()

                loss_val += loss.mean().item()

                # valid_loss_curve.append(loss.mean().item())
                # valid_acc_curve.append(correct_val / total_val)

                type_val_index0 = np.where(target.cpu().numpy() == 0.0)
                total0_val += len(type_val_index0[0])
                for index in type_val_index0[0]:
                    if predicted[index] == target.long()[index]:
                        correct0_val += 1

                type_val_index1 = np.where(target.cpu().numpy() == 1.0)
                total1_val += len(type_val_index1[0])
                for index in type_val_index1[0]:
                    if predicted[index] == target.long()[index]:
                        correct1_val += 1

                type_val_index2 = np.where(target.cpu().numpy() == 2.0)
                total2_val += len(type_val_index2[0])
                for index in type_val_index2[0]:
                    if predicted[index] == target.long()[index]:
                        correct2_val += 1

                type_val_index3 = np.where(target.cpu().numpy() == 3.0)
                total3_val += len(type_val_index3[0])
                for index in type_val_index3[0]:
                    if predicted[index] == target.long()[index]:
                        correct3_val += 1

                type_val_index4 = np.where(target.cpu().numpy() == 4.0)
                total4_val += len(type_val_index4[0])
                for index in type_val_index4[0]:
                    if predicted[index] == target.long()[index]:
                        correct4_val += 1

                type_val_index5 = np.where(target.cpu().numpy() == 5.0)
                total5_val += len(type_val_index5[0])
                for index in type_val_index5[0]:
                    if predicted[index] == target.long()[index]:
                        correct5_val += 1

                # valid_loss_curve.append(loss.mean().item())
                # valid_acc_curve.append(correct_val / total_val)

            valid_loss_curve.append(loss_val / valid_loader.__len__())
            valid_acc_curve.append(correct_val / total_val)
            print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3} Loss:{:.4f} Acc:{:.2%}".format(epoch,
                                                                                                        args.epoch,
                                                                                                        j + 1,
                                                                                                        len(
                                                                                                            valid_loader),
                                                                                                        loss_val / valid_loader.__len__(),
                                                                                                        correct_val / total_val))

            if total0_val == 0:
                valid_acc_curve0.append(0.0)
            else:
                valid_acc_curve0.append(correct0_val / total0_val)

            if total1_val == 0:
                valid_acc_curve1.append(0.0)
            else:
                valid_acc_curve1.append(correct1_val / total1_val)

            if total2_val == 0:
                valid_acc_curve2.append(0.0)
            else:
                valid_acc_curve2.append(correct2_val / total2_val)

            if total3_val == 0:
                valid_acc_curve3.append(0.0)
            else:
                valid_acc_curve3.append(correct3_val / total3_val)

            if total4_val == 0:
                valid_acc_curve4.append(0.0)
            else:
                valid_acc_curve4.append(correct4_val / total4_val)

            if total5_val == 0:
                valid_acc_curve5.append(0.0)
            else:
                valid_acc_curve5.append(correct5_val / total5_val)

with open("./results/train_neg0.txt", mode='w', encoding='utf-8') as f0:
    for val in train_acc_curve0:
        f0.write(str(val))
        f0.write('\n')

with open("./results/train_neg1.txt", mode='w', encoding='utf-8') as f1:
    for val in train_acc_curve1:
        f1.write(str(val))
        f1.write('\n')

with open("./results/train_neg2.txt", mode='w', encoding='utf-8') as f2:
    for val in train_acc_curve2:
        f2.write(str(val))
        f2.write('\n')

with open("./results/train_neg3.txt", mode='w', encoding='utf-8') as f3:
    for val in train_acc_curve3:
        f3.write(str(val))
        f3.write('\n')

with open("./results/train_neg4.txt", mode='w', encoding='utf-8') as f4:
    for val in train_acc_curve4:
        f4.write(str(val))
        f4.write('\n')

with open("./results/train_pos.txt", mode='w', encoding='utf-8') as f5:
    for val in train_acc_curve5:
        f5.write(str(val))
        f5.write('\n')

with open("./results/valid_neg0.txt", mode='w', encoding='utf-8') as f6:
    for val in valid_acc_curve0:
        f6.write(str(val))
        f6.write('\n')

with open("./results/valid_neg1.txt", mode='w', encoding='utf-8') as f7:
    for val in valid_acc_curve1:
        f7.write(str(val))
        f7.write('\n')

with open("./results/valid_neg2.txt", mode='w', encoding='utf-8') as f8:
    for val in valid_acc_curve2:
        f8.write(str(val))
        f8.write('\n')

with open("./results/valid_neg3.txt", mode='w', encoding='utf-8') as f9:
    for val in valid_acc_curve3:
        f9.write(str(val))
        f9.write('\n')

with open("./results/valid_neg4.txt", mode='w', encoding='utf-8') as f10:
    for val in valid_acc_curve4:
        f10.write(str(val))
        f10.write('\n')

with open("./results/valid_pos.txt", mode='w', encoding='utf-8') as f11:
    for val in valid_acc_curve5:
        f11.write(str(val))
        f11.write('\n')

with open("./results/train_loss.txt", mode='w', encoding='utf-8') as f12:
    for val in train_loss_curve:
        f12.write(str(val))
        f12.write('\n')

with open("./results/valid_loss.txt", mode='w', encoding='utf-8') as f13:
    for val in valid_loss_curve:
        f13.write(str(val))
        f13.write('\n')

with open("./results/train_target.txt", mode='w', encoding='utf-8') as f14:
    for val in train_real_label:
        f14.write(str(val))
        f14.write('\n')

with open("./results/train_pre.txt", mode='w', encoding='utf-8') as f15:
    for val in train_pre_label:
        f15.write(str(val))
        f15.write('\n')

with open("./results/valid_target.txt", mode='w', encoding='utf-8') as f16:
    for val in valid_real_label:
        f16.write(str(val))
        f16.write('\n')

with open("./results/valid_pre.txt", mode='w', encoding='utf-8') as f17:
    for val in valid_pre_label:
        f17.write(str(val))
        f17.write('\n')
with open("./results/select_idx.txt", mode='w', encoding='utf-8') as f18:
    for arr in select_idx_list:
        for val in arr:
            f18.write(str(val))
            f18.write(' ')
        f18.write('\n')
with open("./results/train_acc.txt", mode='w', encoding='utf-8') as f19:
    for val in train_acc_curve:
        f19.write(str(val))
        f19.write('\n')
with open("./results/valid_acc.txt", mode='w', encoding='utf-8') as f20:
    for val in valid_acc_curve:
        f20.write(str(val))
        f20.write('\n')
f21 = open("./results/key_layer_weight.txt", mode='w', encoding='utf-8')
f22 = open("./results/key_layer_bias.txt", mode='w', encoding='utf-8')
f23 = open("./results/query_layer_weight.txt", mode='w', encoding='utf-8')
f24 = open("./results/query_layer_bias.txt", mode='w', encoding='utf-8')
f25 = open("./results/value_layer_weight.txt", mode='w', encoding='utf-8')
f26 = open("./results/value_layer_bias.txt", mode='w', encoding='utf-8')
for name, parameter in deep_model.named_parameters():
    # print(name)
    if name == 'tcn.network.0.key_layer.weight':
        # print(parameter)
        for arr in parameter.cpu():
            for val in arr.detach().cpu().numpy():
                f21.write(str(val))
                f21.write(' ')
            f21.write('\n')
    if name == 'tcn.network.0.key_layer.bias':
        for val in parameter.detach().cpu().numpy():
            f22.write(str(val))
            f22.write(' ')
        f22.write('\n')
    if name == 'tcn.network.0.query_layer.weight':
        for arr in parameter.cpu():
            for val in arr.detach().cpu().numpy():
                f23.write(str(val))
                f23.write(' ')
            f23.write('\n')
    if name == 'tcn.network.0.query_layer.bias':
        for val in parameter.detach().cpu().numpy():
            f24.write(str(val))
            f24.write(' ')
        f24.write('\n')
    if name == 'tcn.network.0.value_layer.weight':
        for arr in parameter.cpu():
            for val in arr.detach().cpu().numpy():
                f25.write(str(val))
                f25.write(' ')
            f25.write('\n')
    if name == 'tcn.network.0.value_layer.bias':
        for val in parameter.detach().cpu().numpy():
            f26.write(str(val))
            f26.write(' ')
        f26.write('\n')

f26.close()
f25.close()
f24.close()
f23.close()
f22.close()
f21.close()
