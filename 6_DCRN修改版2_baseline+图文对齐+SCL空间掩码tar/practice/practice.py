import numpy as np
import torch
import random

# data = {13:2, 2:4, 4:8}
# data = sorted(list(data)) # 输出的是key值
# print(data)

# list1 = [1,2,3,4,5,6,7,8,9]
# samples = random.sample(list1, len(list1))
# print(samples)
# random.shuffle(samples)
# print(samples)

# data = np.arange(12)
# data = data.reshape(-1, 2, 3)
# print(data.shape)
# print(type(data.shape))
# print(data.shape[0])
# print(data.shape[2])
# print(data.shape[1])
# print(data)

import numpy as np
import random

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

# HT dropout = 0.1   average OA: 76.05 +- 2.05
# acc = [77.34688858,73.61646306,78.37992949,76.80577191,79.29818808,73.94441256,75.67434615,77.57645323,72.88677544,74.99385095]
# HT dropout = 0.3   average OA: 76.33 +- 1.87
# acc = [76.9123555 ,73.23112241,78.05198   ,77.50266459,79.76551611,74.65770271,76.10887923,77.42887595,74.59211281,75.0266459 ]
# HT dropout = 0.5   average OA: 76.08 +- 1.73
# acc = [76.20726408,74.86267115,77.28129868,76.06788555,79.31458555,74.87906862,75.01844716,78.00278757,72.89497417,76.30564893]
# HT dropout = 0.9   average OA: 75.80 +- 1.98
# acc = [76.24005903,72.42764614,77.94539641,78.01918505,76.98614413,72.41124867,75.6415512 ,76.75657949,74.33795196,77.19931131]

# IP dropout = 0.1   average OA: 81.45 +- 1.65
# acc = [83.12518438,82.25980922,79.40800472,82.81050251,79.55551185,80.25371226,79.13265808,82.24997542,81.75828498,83.99055954]
# IP dropout = 0.3  average OA: 81.58 +- 1.73
# acc = [83.65621005,83.03668011,80.22421084,82.75149966,79.78168945,79.60468089,78.86714525,82.32864588,81.72878356,83.84305241]
# IP dropout = 0.7 average OA: 81.41 +- 1.59
# acc = [83.144852  ,83.06618153,79.7226866 ,82.90884059,80.10620513,80.38155178,78.86714525,82.53515587,80.32254892,83.08584915]
# IP dropout = 0.9  average OA: 77.49 +- 1.88
# acc = [80.32254892,76.11367883,77.07739207,76.32018881,76.29068738,76.79221162,76.87088209,75.27780509,78.33611958,81.47310453]

# SA dropout = 0.1  average OA: 94.05 +- 0.93
# acc = [95.07853984,92.70661807,94.62339729,92.92863883,93.38748173,93.2801717 ,93.52994505,95.42452219,94.80471424,94.68815334]

# SA dropout = 0.3 average OA: 94.25 +- 0.81
# acc = [95.45042461,93.48184055,94.27741494,92.91938796,93.63910526,93.81302152,93.89997965,95.49852911,94.93977687,94.54383985]

# SA dropout = 0.7 average OA: 94.22 +- 0.75
# acc = [95.39306925,93.07850284,94.34217099,93.20986512,93.66130733,93.80377065,94.09794816,94.96197895,95.01563396,94.63819867]

# SA dropout = 0.9 average OA: 93.68 +- 0.96
# acc = [93.77046754,91.72047586,93.39118208,92.34953468,94.06834539,93.84262429,94.15530352,95.30796129,93.95363466,94.26261355]

# LK  average OA: 96.10 +- 2.12
acc = [97.60338782,95.85568492,95.93099165,96.41461733,97.42734612,97.72710602,90.39056808,94.89430163,96.83124936,97.93591104]

OAMean = np.mean(acc)
OAStd = np.std(acc)
print("average OA: " + "{:.2f}".format(OAMean) + " +- " + "{:.2f}".format(OAStd))

# import torch
# from transformers import BertTokenizer, BertModel
#
# # 加载预训练的BERT模型和分词器
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
#
# # 将标签列表转换为Tensor
# labels_tensor = torch.tensor(labels_tar)
#
# # 编码单个单词的特征向量
# def encode_word(word):
#     # 使用BERT的分词器对单词进行编码
#     input_ids = tokenizer.encode(word, add_special_tokens=False)
#     input_ids_tensor = torch.tensor(input_ids).unsqueeze(0)  # 添加批次维度
#
#     # 使用BERT模型获取特征向量
#     with torch.no_grad():
#         outputs = model(input_ids_tensor)
#         encoded_word = torch.mean(outputs.last_hidden_state, dim=1)  # 取平均作为特征向量
#
#     return encoded_word
#
# # 编码整个标签的特征向量并计算平均值
# def encode_label(label):
#     words = label.split("-")  # 使用"-"分割多个单词
#     encoded_words = [encode_word(word) for word in words]  # 对每个单词进行编码
#     encoded_label = torch.mean(torch.stack(encoded_words), dim=0)  # 计算编码后的单词向量的平均值
#     return encoded_label
#
# # 对标签进行编码并计算平均值
# encoded_labels = [encode_label(label) for label in labels_tar]
# semantic_vectors = torch.stack(encoded_labels)
#
# print(semantic_vectors)


# 加载目标域数据
# import os
# from utils import utils

# test_data = os.path.join('../../../datasets','houston2013/Houston.mat')  # (349, 1905, 144)
# test_label = os.path.join('../../../datasets','houston2013/Houston_gt.mat')  # (349, 1905)

# test_data = os.path.join('../../../datasets','Houston/data.mat')  # (349, 1905, 144)
# test_label = os.path.join('../../../datasets','Houston/mask_test.mat')  # (349, 1905)
#
# Data_Band_Scaler, GroundTruth = utils.load_data(test_data, test_label)
#
# data = GroundTruth
# mask = np.unique(data)
# labeled_pixels = 0
# sum_pixels = 0
# tmp = {}
# for v in mask:
#     tmp[v] = np.sum(data == v)
#     if v > 0:
#         labeled_pixels += tmp[v]
#     sum_pixels += tmp[v]
# print(mask) # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]
# print(tmp) # {0: 649816, 1: 1251, 2: 1254, 3: 697, 4: 1244, 5: 1242, 6: 325, 7: 1268, 8: 1244, 9: 1252, 10: 1227, 11: 1235, 12: 1233, 13: 469, 14: 428, 15: 660}
# print(labeled_pixels) # 15029
# print(sum_pixels)
#
# print("OK")

# mask_train
# {0: 662013, 1: 198, 2: 190, 3: 192, 4: 188, 5: 186, 6: 182, 7: 196, 8: 191, 9: 193, 10: 191, 11: 181, 12: 192, 13: 184, 14: 181, 15: 187}
# 2832
# 664845

# mask_test
# {0: 652648, 1: 1053, 2: 1064, 3: 505, 4: 1056, 5: 1056, 6: 143, 7: 1072, 8: 1053, 9: 1059, 10: 1036, 11: 1054, 12: 1041, 13: 285, 14: 247, 15: 473}
# 12197  81.15643%
# 664845









class MetaTrainDataset(Dataset):
    def __init__(self, image_datas):
        self.image_datas = image_datas

    def __len__(self):
        return len(self.image_datas)

    def __getitem__(self, idx):
        image = self.image_datas[idx]
        return image

def get_metatrain_data_loader():
    domain_specific_metatrain_data = [i for i in range(10)]
    dataset = MetaTrainDataset(domain_specific_metatrain_data)
    loader = DataLoader(dataset, batch_size = 4)

    return loader

# loader = get_metatrain_data_loader()
# episode = 100
# source_iter = iter(loader)

# 这种写法是正确的
# for i in range(episode):
#     try:
#         source_data = source_iter.next()
#         print(source_data)
#     except Exception as err:
#         source_iter = iter(loader)
#         source_data = source_iter.next()
#         print(source_data)

# 错误写法
# train_cl = loader.__iter__().next() # 之前程序写错了！！！！！！！！！！！而且shuffle应该设为true
# print(train_cl) # 当loader的shuffle=false时，全是tensor([0, 1])；=true时候，会输出不一样的



# for data in loader:
#     print(data)
# tensor([0, 1])
# tensor([2, 3])
# tensor([4, 5])
# tensor([6, 7])
# tensor([8, 9])

# np1 = np.random.uniform(1,10,(3,2))
# np2 = np.random.uniform(1,10,(3,2))
# list1 = []
# list1.append(np1)
# list1.append(np2)
# np3 = np.array(list1)
# print(np3)
# print(np3.shape)

# # Import required package
# import numpy as np
#
# # Creating a Dictionary
# # with Integer Keys
# dict = {1: 'Geeks',
#         2: 'For',
#         3: 'Geeks'}
#
# # to return a group of the key-value
# # pairs in the dictionary
# result = dict.items()
# for i in result:
#     print(i)
#
# # Convert object to a list
# data = list(result)
#
# # Convert list to an array
# numpyArray = np.array(data)
#
# # print the numpy array
# print(numpyArray)

# list1 = [1,2,3]
# list2 = [4,5]
# list1 += list2
# print(list1)

# list = []
# a = np.array([6,7,8,9])
# b = np.array([8,9,19,19])
# list.append(a)
# list.append(b)
# print(list)
# list = np.array(list)
# print(list)
# print(list.shape)

def gaussian_noise(image, mean=0, sigma=0.15):
    # mean=0, sigma=0.15
    # mean=0, sigma=0.1
    # mean=0, sigma=0.05
    # 1.可以采用同样的mean和sigma增强，因为即使相同的种子，在一次迭代中顺序随机出的数字也不同
    # 2.也可以采用不同的mean和sigma增强

    """
    添加高斯噪声
    :param image:原图
    param mean:均值
    :param sigma:标准差 值越大，噪声越多
    :return:噪声处理后的图片
    """

    image = np.asarray(image, dtype=np.float32)
    max = np.max(image)
    min = np.min(image)
    length = max - min
    image = (image - min) / length
    noise = np.random.normal(mean, sigma, image.shape).astype(dtype=np.float32)  # 产生高斯噪声
    output = image + noise  # 将噪声和图片叠加
    output = output * length + min
    print("output.max = ", np.max(output))
    print("output.min = ", np.min(output))
    print("output.mean = ", np.mean(output))
    # output = np.clip(output, 0, 1)
    # output = np.uint8(output * length)
    return output




# def gaussian_noise(image, mean=0, sigma=0.15):
#     # mean=0, sigma=0.15
#     # mean=0, sigma=0.1
#     # mean=0, sigma=0.05
#
#     """
#     添加高斯噪声
#     :param image:原图
#     param mean:均值
#     :param sigma:标准差 值越大，噪声越多
#     :return:噪声处理后的图片
#     """
#     # in code ,np.random is up to seed, so need different sigma value , example sigma = 0.1 0.05 0.15
#
#     image = np.asarray(image, dtype=np.float32)
#     max = np.max(image)
#     min = np.min(image)
#     length = max - min
#     print("max = ", max)
#     print("min = ", min)
#     print("length = ", length)
#     print("image.mean = ", np.mean(image))
#     image = (image - min) / length
#     # image = np.asarray((image - min) / length, dtype=np.float32)  # 图片灰度标准化 0-1
#
#     noise1 = np.random.normal(mean, sigma, image.shape).astype(dtype=np.float32)  # 产生高斯噪声
#     noise2 = np.random.normal(mean, sigma, image.shape).astype(dtype=np.float32)  # 产生高斯噪声
#     output1 = image + noise1  # 将噪声和图片叠加
#     output2 = image + noise2
#     output1 = output1 * length + min
#     output2 = output2 * length + min
#     # output = np.clip(output, 0, 1)
#     # output = np.uint8(output * length)
#     print("output1.max = ", np.max(output1))
#     print("output1.min = ", np.min(output1))
#     print("output1.mean = ", np.mean(output1))
#     print("output2.max = ", np.max(output2))
#     print("output2.min = ", np.min(output2))
#     print("output2.mean = ", np.mean(output2))
#     return output1, output2

# torch.manual_seed(0)
# np.random.seed(0)
# supports_src = torch.randn([3,6,3,3]).to(0)
# supports_src = supports_src * 30
# # print(supports_src.device)
# # print(type(supports_src))
# # print(supports_src)
# supports_src1 = torch.FloatTensor(gaussian_noise(supports_src.data.cpu())).to(0)
# supports_src2 = torch.FloatTensor(gaussian_noise(supports_src.data.cpu())).to(0)




# print("-------------------------------------------------------------")
# print(supports_src.shape)
# print(supports_src.device)
# print(type(supports_src))
# print(supports_src1)
# print(supports_src2)

# np.random.seed(0)
# a = np.random.randn(10)
# b = np.random.randn(10)
# c = np.random.randn(10)
# print("a = ", a)
# print("b = ", b)
# print("c = ", c)
# 可以看到三次的结果是不一样的，np.random.seed(0)是保证每次运行程序时和上次的结果一样，第一个np.random.randn(10)会随机出的数字是确定等于上次的，第二次会随机出的数字是确定等于上次的，但不等于第一个
# output
# a =  [ 1.76405235  0.40015721  0.97873798  2.2408932   1.86755799 -0.97727788
#   0.95008842 -0.15135721 -0.10321885  0.4105985 ]
# b =  [ 0.14404357  1.45427351  0.76103773  0.12167502  0.44386323  0.33367433
#   1.49407907 -0.20515826  0.3130677  -0.85409574]
# c =  [-2.55298982  0.6536186   0.8644362  -0.74216502  2.26975462 -1.45436567
#   0.04575852 -0.18718385  1.53277921  1.46935877]



# source
# CH  include background. data.max =  15133.0  data.min =  0.0 data_scaler.max =  88.62107407294518 data_scaler.min =  -2.511908360528769
# CH  data_train.max =  32.416466  data_train.min =  -2.352965
# UP  data.max =  8000  data.min =  0         Data_Band_Scaler.max =  15.918664053448387   Data_Band_Scaler.min =  -2.8023924636727973
# Salinas  data.max =  9207  data.min =  -11  Data_Band_Scaler.max =  222.7637925583525    Data_Band_Scaler.min =  -6.272498288061055
# IP  data.max =  9604  data.min =  955        Data_Band_Scaler.max =  8.992655544253978   Data_Band_Scaler.min =  -7.642090293989591




