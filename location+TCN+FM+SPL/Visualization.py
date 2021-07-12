import matplotlib.pyplot as plt

train_neg0_acc_y = []
with open("./results/train_neg0.txt", mode='r', encoding='utf-8') as f0:
    lines = f0.readlines()
    for line in lines:
        val = eval(line.strip())
        train_neg0_acc_y.append(val)

train_neg1_acc_y = []
with open("./results/train_neg1.txt", mode='r', encoding='utf-8') as f1:
    lines = f1.readlines()
    for line in lines:
        val = eval(line.strip())
        train_neg1_acc_y.append(val)

train_neg2_acc_y = []
with open("./results/train_neg2.txt", mode='r', encoding='utf-8') as f2:
    lines = f2.readlines()
    for line in lines:
        val = eval(line.strip())
        train_neg2_acc_y.append(val)

train_neg3_acc_y = []
with open("./results/train_neg3.txt", mode='r', encoding='utf-8') as f3:
    lines = f3.readlines()
    for line in lines:
        val = eval(line.strip())
        train_neg3_acc_y.append(val)

train_neg4_acc_y = []
with open("./results/train_neg4.txt", mode='r', encoding='utf-8') as f4:
    lines = f4.readlines()
    for line in lines:
        val = eval(line.strip())
        train_neg4_acc_y.append(val)

train_pos_acc_y = []
with open("./results/train_pos.txt", mode='r', encoding='utf-8') as f5:
    lines = f5.readlines()
    for line in lines:
        val = eval(line.strip())
        train_pos_acc_y.append(val)

plt.subplot(231)
plt.plot(range(len(train_neg0_acc_y)), train_neg0_acc_y, label='neg0 train acc')
plt.title("the neg0 train acc on training")
plt.legend(loc='best')
plt.ylabel('acc value')
plt.xlabel('Iteration')

plt.subplot(232)
plt.plot(range(len(train_neg1_acc_y)), train_neg1_acc_y, label='neg1 train acc')
plt.title("the neg1 train acc on training")
plt.legend(loc='best')
plt.ylabel('acc value')
plt.xlabel('Iteration')

plt.subplot(233)
plt.plot(range(len(train_neg2_acc_y)), train_neg2_acc_y, label='neg2 train acc')
plt.title("the neg2 train acc on training")
plt.legend(loc='best')
plt.ylabel('acc value')
plt.xlabel('Iteration')

plt.subplot(234)
plt.plot(range(len(train_neg3_acc_y)), train_neg3_acc_y, label='neg3 train acc')
plt.title("the neg3 train acc on training")
plt.legend(loc='best')
plt.ylabel('acc value')
plt.xlabel('Iteration')

plt.subplot(235)
plt.plot(range(len(train_neg4_acc_y)), train_neg4_acc_y, label='neg4 train acc')
plt.title("the neg4 train acc on training")
plt.legend(loc='best')
plt.ylabel('acc value')
plt.xlabel('Iteration')

plt.subplot(236)
plt.plot(range(len(train_pos_acc_y)), train_pos_acc_y, label='pos train acc')
plt.title("the pos train acc on training")
plt.legend(loc='best')
plt.ylabel('acc value')
plt.xlabel('Iteration')

plt.show()