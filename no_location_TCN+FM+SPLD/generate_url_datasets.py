import numpy as np
import torch
import pandas as pd

# ---------------------------加载训练的词向量-----------------------------
word_vec_dict = dict()
with open(file="./vec_dict/url_vec_no_location1216.txt", mode='r', encoding='utf-8') as f1:
    lines = f1.readlines()
    for line in lines:
        url_vec_line = line.split()
        key = url_vec_line[0]
        value = []
        for word_val in url_vec_line[1:]:
            value.append(eval(word_val))
        word_vec_dict[key] = value
url_tokens_num_list = []

# **********************************************************************************************************#
# ---------------------------构造第一类url(doamin存在于neeloc中)的deepfeature,classes = 0--------------------
all_url_vec_type0 = []
with open(file="./neg_url_tokens/domain_exist_at_nec_tokens_without_location.txt", mode='r', encoding='utf-8') as f2:
    lines = f2.readlines()[:373946]
    for line in lines:
        words_list = line.split()
        single_url_vec = []
        for word in words_list:
            if word in word_vec_dict.keys():
                single_url_vec.append(word_vec_dict[word])
            else:
                single_url_vec.append([0.0] * 100)
        for i in range(len(single_url_vec), 39):
            single_url_vec.append([0.0] * 100)
        single_url_vec_arr = np.array(single_url_vec).reshape(1, -1)
        all_url_vec_type0.append(single_url_vec_arr)

neg_deep_feature0 = torch.tensor(all_url_vec_type0, dtype=torch.float32).view(373946,
                                                                              -1)  # [373946,3900]

# ---------------------------构造第一类url(doamin存在于neeloc中)的widefeature,classes = 0--------------------

neg_wide_feature0 = pd.read_csv("./neg_url_widefeature/domain_exist_at_net_wideFeature.csv", encoding='utf-8')
neg_wide_feature0 = neg_wide_feature0.values[:373946, :]
neg_wide_feature0 = torch.tensor(neg_wide_feature0, dtype=torch.float32)  # [373946,34]
# print(neg_wide_feature.dtype)

# ---------------------------构造第一类url(doamin存在于neeloc中)的feature,classes = 0--------------------

neg_url_feature0 = torch.cat([neg_wide_feature0, neg_deep_feature0], dim=1)  # [373946,3934]
# print(neg_url_feature.shape)

# ---------------------------构造第一类url(doamin存在于neeloc中)的所属累别,classes = 0--------------------
neg_url_classes0 = torch.zeros((373946, 1), dtype=torch.float32)
# print(neg_url_label.shape)

# ---------------------------构造第一类url(doamin存在于neeloc中)的target列(即正样本还是负样本),classes = 0--------
neg_url_label0 = torch.zeros((373946, 1), dtype=torch.float32)
# print(neg_url_label.shape)

# ---------------------------构造第一类url(doamin存在于neeloc中)数据，包括所属类别和target列,classes = 0--------
neg_url_feature00 = torch.cat([neg_url_feature0, neg_url_classes0], dim=1)
neg_url_feature000 = torch.cat([neg_url_feature00, neg_url_label0], dim=1)
# print(neg_url_feature000.shape)


# ---------------------------构造第二类url(doamin是ip)的deepfeature,classes = 1--------------------
all_url_vec_type1 = []
with open(file="./neg_url_tokens/domain_is_ip_tokens_without_location.txt", mode='r', encoding='utf-8') as f3:
    lines = f3.readlines()
    for line in lines:
        words_list = line.split()
        single_url_vec = []
        for word in words_list:
            if word in word_vec_dict.keys():
                single_url_vec.append(word_vec_dict[word])
            else:
                single_url_vec.append([0.0] * 100)
        for i in range(len(single_url_vec), 39):
            single_url_vec.append([0.0] * 100)
        single_url_vec_arr = np.array(single_url_vec).reshape(1, -1)
        all_url_vec_type1.append(single_url_vec_arr)

neg_deep_feature1 = torch.tensor(all_url_vec_type1, dtype=torch.float32).view(16790, -1)  # [16790,3900]

# ---------------------------构造第二类url(doamin存在于neeloc中)的widefeature,classes = 1--------------------

neg_wide_feature1 = pd.read_csv("./neg_url_widefeature/domain_is_ip_wideFeature.csv", encoding='utf-8')
neg_wide_feature1 = neg_wide_feature1.values
neg_wide_feature1 = torch.tensor(neg_wide_feature1, dtype=torch.float32)  # [16790,34]
# print(neg_wide_feature.dtype)

# ---------------------------构造第二类url(doamin存在于neeloc中)的feature,classes = 1--------------------

neg_url_feature1 = torch.cat([neg_wide_feature1, neg_deep_feature1], dim=1)  # [16790,3934]
# print(neg_url_feature.shape)

# ---------------------------构造第二类url(doamin存在于neeloc中)的所属累别,classes = 1--------------------
neg_url_classes1 = torch.ones((16790, 1), dtype=torch.float32)
# print(neg_url_label.shape)

# ---------------------------构造第二类url(doamin存在于neeloc中)的target列(即正样本还是负样本),classes = 1--------
neg_url_label1 = torch.zeros((16790, 1), dtype=torch.float32)
# print(neg_url_label.shape)

# ---------------------------构造第二类url(doamin存在于neeloc中)数据，包括所属类别和target列,classes = 1--------
neg_url_feature11 = torch.cat([neg_url_feature1, neg_url_classes1], dim=1)
neg_url_feature111 = torch.cat([neg_url_feature11, neg_url_label1], dim=1)
# print(neg_url_feature111.shape)


# ---------------------------构造第三类url(doamin存在于net中和其他地方)的deepfeature,classes = 2--------------------
all_url_vec_type2 = []
with open(file="./neg_url_tokens/real_domain_exist_at_nec_and_others_tokens_without_location.txt", mode='r',
          encoding='utf-8') as f4:
    lines = f4.readlines()
    for line in lines:
        words_list = line.split()
        single_url_vec = []
        for word in words_list:
            if word in word_vec_dict.keys():
                single_url_vec.append(word_vec_dict[word])
            else:
                single_url_vec.append([0.0] * 100)
        for i in range(len(single_url_vec), 39):
            single_url_vec.append([0.0] * 100)
        single_url_vec_arr = np.array(single_url_vec).reshape(1, -1)
        all_url_vec_type2.append(single_url_vec_arr)

neg_deep_feature2 = torch.tensor(all_url_vec_type2, dtype=torch.float32).view(18128, -1)  # [18128,3900]

# ---------------------------构造第三类url(doamin存在于net中和其他地方)的widefeature,classes = 2--------------------

neg_wide_feature2 = pd.read_csv("./neg_url_widefeature/real_domain_exist_at_nec_and_others_wideFeature.csv",
                                encoding='utf-8')
neg_wide_feature2 = neg_wide_feature2.values
neg_wide_feature2 = torch.tensor(neg_wide_feature2, dtype=torch.float32)  # [18128,34]
# print(neg_wide_feature.dtype)

# ---------------------------构造第三类url(doamin存在于net中和其他地方)的feature,classes = 2--------------------

neg_url_feature2 = torch.cat([neg_wide_feature2, neg_deep_feature2], dim=1)  # [18128,3934]
# print(neg_url_feature.shape)

# ---------------------------构造第三类url(doamin存在于net中和其他地方)的所属累别,classes = 2--------------------
neg_url_classes2 = 2 * torch.ones((18128, 1), dtype=torch.float32)
# print(neg_url_label.shape)

# ---------------------------构造第三类url(doamin存在于net中和其他地方)的target列(即正样本还是负样本),classes = 2--------
neg_url_label2 = torch.zeros((18128, 1), dtype=torch.float32)
# print(neg_url_label.shape)

# ---------------------------构造第三类url(doamin存在于net中和其他地方)数据，包括所属类别和target列,classes = 2--------
neg_url_feature22 = torch.cat([neg_url_feature2, neg_url_classes2], dim=1)
neg_url_feature222 = torch.cat([neg_url_feature22, neg_url_label2], dim=1)

# ---------------------------构造第四类url(doamin存在于其他地方)的deepfeature,classes = 3--------------------
all_url_vec_type3 = []
with open(file="./neg_url_tokens/real_domain_exist_at_others_tokens_without_location.txt", mode='r',
          encoding='utf-8') as f5:
    lines = f5.readlines()
    for line in lines:
        words_list = line.split()
        single_url_vec = []
        for word in words_list:
            if word in word_vec_dict.keys():
                single_url_vec.append(word_vec_dict[word])
            else:
                single_url_vec.append([0.0] * 100)
        for i in range(len(single_url_vec), 39):
            single_url_vec.append([0.0] * 100)
        single_url_vec_arr = np.array(single_url_vec).reshape(1, -1)
        all_url_vec_type3.append(single_url_vec_arr)

neg_deep_feature3 = torch.tensor(all_url_vec_type3, dtype=torch.float32).view(2705, -1)  # [2705,3900]

# ---------------------------构造第四类url(doamin存在于其他地方)的widefeature,classes = 3--------------------

neg_wide_feature3 = pd.read_csv("./neg_url_widefeature/real_domain_exist_at_others_wideFeature.csv",
                                encoding='utf-8')
neg_wide_feature3 = neg_wide_feature3.values
neg_wide_feature3 = torch.tensor(neg_wide_feature3, dtype=torch.float32)  # [2705,34]
# print(neg_wide_feature.dtype)

# ---------------------------构造第四类url(doamin存在于其他地方)的feature,classes = 3--------------------

neg_url_feature3 = torch.cat([neg_wide_feature3, neg_deep_feature3], dim=1)  # [2705,3934]
# print(neg_url_feature.shape)

# ---------------------------构造第四类url(doamin存在于其他地方)的所属累别,classes = 3--------------------
neg_url_classes3 = 3 * torch.ones((2705, 1), dtype=torch.float32)
# print(neg_url_label.shape)

# ---------------------------构造第四类url(doamin存在于其他地方)的target列(即正样本还是负样本),classes = 3--------
neg_url_label3 = torch.zeros((2705, 1), dtype=torch.float32)
# print(neg_url_label.shape)

# ---------------------------构造第四类url(doamin存在于其他地方)数据，包括所属类别和target列,classes = 3--------
neg_url_feature33 = torch.cat([neg_url_feature3, neg_url_classes3], dim=1)
neg_url_feature333 = torch.cat([neg_url_feature33, neg_url_label3], dim=1)

# ---------------------------构造第五类url(doamin不明显)的deepfeature,classes = 4--------------------
all_url_vec_type4 = []
with open(file="./neg_url_tokens/url_no_obvious_domain_tokens_without_location.txt", mode='r', encoding='utf-8') as f6:
    lines = f6.readlines()
    for line in lines:
        words_list = line.split()
        single_url_vec = []
        for word in words_list:
            if word in word_vec_dict.keys():
                single_url_vec.append(word_vec_dict[word])
            else:
                single_url_vec.append([0.0] * 100)
        for i in range(len(single_url_vec), 39):
            single_url_vec.append([0.0] * 100)
        single_url_vec_arr = np.array(single_url_vec).reshape(1, -1)
        all_url_vec_type4.append(single_url_vec_arr)

neg_deep_feature4 = torch.tensor(all_url_vec_type4, dtype=torch.float32).view(68431, -1)  # [68431,3900]

# ---------------------------构造第五类url(doamin不明显)的widefeature,classes = 4--------------------

neg_wide_feature4 = pd.read_csv("./neg_url_widefeature/url_no_obvious_domain_wideFeature.csv",
                                encoding='utf-8')
neg_wide_feature4 = neg_wide_feature4.values
neg_wide_feature4 = torch.tensor(neg_wide_feature4, dtype=torch.float32)  # [68431,34]
# print(neg_wide_feature.dtype)

# ---------------------------构造第五类url(ddoamin不明显)的feature,classes = 4--------------------

neg_url_feature4 = torch.cat([neg_wide_feature4, neg_deep_feature4], dim=1)  # [68431,3934]
# print(neg_url_feature.shape)

# ---------------------------构造第五类url(ddoamin不明显)的所属累别,classes = 4--------------------
neg_url_classes4 = 4 * torch.ones((68431, 1), dtype=torch.float32)
# print(neg_url_label.shape)

# ---------------------------构造第五类url(ddoamin不明显)的target列(即正样本还是负样本),classes = 4--------
neg_url_label4 = torch.zeros((68431, 1), dtype=torch.float32)
# print(neg_url_label.shape)

# ---------------------------构造第五类url(ddoamin不明显)数据，包括所属类别和target列,classes = 4--------
neg_url_feature44 = torch.cat([neg_url_feature4, neg_url_classes4], dim=1)
neg_url_feature444 = torch.cat([neg_url_feature44, neg_url_label4], dim=1)

# ---------------------------构造第六类url(正类)的deepfeature,classes = 5--------------------
all_url_vec_type5 = []
with open(file="./pos_url_tokens/pos_url_tokens_without_location.txt", mode='r', encoding='utf-8') as f7:
    lines = f7.readlines()
    for line in lines:
        words_list = line.split()
        single_url_vec = []
        for word in words_list:
            if word in word_vec_dict.keys():
                single_url_vec.append(word_vec_dict[word])
            else:
                single_url_vec.append([0.0] * 100)
        for i in range(len(single_url_vec), 39):
            single_url_vec.append([0.0] * 100)
        single_url_vec_arr = np.array(single_url_vec).reshape(1, -1)
        all_url_vec_type5.append(single_url_vec_arr)

neg_deep_feature5 = torch.tensor(all_url_vec_type5, dtype=torch.float32).view(480000,
                                                                              -1)  # [480000,3900]

# ---------------------------构造第六类url(正类)的widefeature,classes = 5--------------------

neg_wide_feature5 = pd.read_csv("./pos_url_widefeature/pos_url_wideFeature.csv",
                                encoding='utf-8')
neg_wide_feature5 = neg_wide_feature5.values
neg_wide_feature5 = torch.tensor(neg_wide_feature5, dtype=torch.float32)  # [480000,34]
# print(neg_wide_feature.dtype)

# ---------------------------构造第六类url(正类)的feature,classes = 5--------------------

neg_url_feature5 = torch.cat([neg_wide_feature5, neg_deep_feature5], dim=1)  # [480000,3934]
# print(neg_url_feature.shape)

# ---------------------------构造第六类url(正类)的所属累别,classes = 5--------------------
neg_url_classes5 = 5 * torch.ones((480000, 1), dtype=torch.float32)
# print(neg_url_label.shape)

# ---------------------------构造第六类url(正类)的target列(即正样本还是负样本),classes = 5--------
neg_url_label5 = torch.ones((480000, 1), dtype=torch.float32)
# print(neg_url_label.shape)

# ---------------------------构造第六类url(正类)数据，包括所属类别和target列,classes = 5--------
neg_url_feature55 = torch.cat([neg_url_feature5, neg_url_classes5], dim=1)
neg_url_feature555 = torch.cat([neg_url_feature55, neg_url_label5], dim=1)

url_data0 = torch.cat([neg_url_feature000, neg_url_feature111], dim=0)
url_data1 = torch.cat([url_data0, neg_url_feature222], dim=0)
url_data2 = torch.cat([url_data1, neg_url_feature333], dim=0)
url_data3 = torch.cat([url_data2, neg_url_feature444], dim=0)
url_data = torch.cat([url_data3, neg_url_feature555], dim=0)

torch.manual_seed(2020)
torch.cuda.manual_seed(2020)
perm = torch.randperm(len(url_data))
url_data = url_data[perm]

url_train_data = url_data[:768000, :]
url_valid_data = url_data[768000:, :]
