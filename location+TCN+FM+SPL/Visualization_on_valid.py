import matplotlib.pyplot as plt

valid_neg0_acc_y = []
with open("./results/valid_neg0.txt", mode='r', encoding='utf-8') as f0:
    lines = f0.readlines()
    for line in lines:
        val = eval(line.strip())
        valid_neg0_acc_y.append(val)

valid_neg1_acc_y = []
with open("./results/valid_neg1.txt", mode='r', encoding='utf-8') as f1:
    lines = f1.readlines()
    for line in lines:
        val = eval(line.strip())
        valid_neg1_acc_y.append(val)

valid_neg2_acc_y = []
with open("./results/valid_neg2.txt", mode='r', encoding='utf-8') as f2:
    lines = f2.readlines()
    for line in lines:
        val = eval(line.strip())
        valid_neg2_acc_y.append(val)

valid_neg3_acc_y = []
with open("./results/valid_neg3.txt", mode='r', encoding='utf-8') as f3:
    lines = f3.readlines()
    for line in lines:
        val = eval(line.strip())
        valid_neg3_acc_y.append(val)

valid_neg4_acc_y = []
with open("./results/valid_neg4.txt", mode='r', encoding='utf-8') as f4:
    lines = f4.readlines()
    for line in lines:
        val = eval(line.strip())
        valid_neg4_acc_y.append(val)

valid_pos_acc_y = []
with open("./results/valid_pos.txt", mode='r', encoding='utf-8') as f5:
    lines = f5.readlines()
    for line in lines:
        val = eval(line.strip())
        valid_pos_acc_y.append(val)

plt.subplot(231)
plt.plot(range(len(valid_neg0_acc_y)), valid_neg0_acc_y, label='neg0 valid acc')
plt.title("the neg0 valid acc on validing")
plt.legend(loc='best')
plt.ylabel('acc value')
plt.xlabel('Iteration')

plt.subplot(232)
plt.plot(range(len(valid_neg1_acc_y)), valid_neg1_acc_y, label='neg1 valid acc')
plt.title("the neg1 valid acc on validing")
plt.legend(loc='best')
plt.ylabel('acc value')
plt.xlabel('Iteration')

plt.subplot(233)
plt.plot(range(len(valid_neg2_acc_y)), valid_neg2_acc_y, label='neg2 valid acc')
plt.title("the neg2 valid acc on validing")
plt.legend(loc='best')
plt.ylabel('acc value')
plt.xlabel('Iteration')

plt.subplot(234)
plt.plot(range(len(valid_neg3_acc_y)), valid_neg3_acc_y, label='neg3 valid acc')
plt.title("the neg3 valid acc on validing")
plt.legend(loc='best')
plt.ylabel('acc value')
plt.xlabel('Iteration')

plt.subplot(235)
plt.plot(range(len(valid_neg4_acc_y)), valid_neg4_acc_y, label='neg4 valid acc')
plt.title("the neg4 valid acc on validing")
plt.legend(loc='best')
plt.ylabel('acc value')
plt.xlabel('Iteration')

plt.subplot(236)
plt.plot(range(len(valid_pos_acc_y)), valid_pos_acc_y, label='pos valid acc')
plt.title("the pos valid acc on validing")
plt.legend(loc='best')
plt.ylabel('acc value')
plt.xlabel('Iteration')

plt.show()