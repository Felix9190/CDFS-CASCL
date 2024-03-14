import numpy as np

import torch
from torchvision import transforms

def flip_augmentation(data): # arrays tuple 0:(7, 7, 103) 1=(7, 7)
    horizontal = np.random.random() > 0.5 # True
    vertical = np.random.random() > 0.5 # False
    if horizontal:
        data = np.fliplr(data)
    if vertical:
        data = np.flipud(data)
    return data

def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
    return alpha * data + beta * noise

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
    # output = np.clip(output, 0, 1)
    # output = np.uint8(output * length)
    return output

# 输入数据形式：(batchsize, channels, patch_size, patch_size) tensor
def Crop_and_resize_batch(data, HalfWidth):
    da = transforms.RandomResizedCrop(2 * HalfWidth + 1, scale = (0.08, 1.0), ratio=(0.75, 1.3333333333333333))
    x = da(data)
    return x

# single image random mask
def random_mask_single_image(input_image, mask_ratio): # input (128, 7, 7)
    patch_size = input_image.shape[1]

    # 生成通道维度的随机掩码张量
    random_mask_channels = torch.rand(input_image.shape[0], 1, 1)

    # 将通道维度的随机掩码张量中大于 mask_ratio 的元素置为 1，小于等于 mask_ratio 的元素置为 0
    random_mask_channels = torch.where(random_mask_channels > mask_ratio, torch.tensor(1.0), torch.tensor(0.0))

    # 生成空间维度的随机掩码张量
    random_mask_spatial = torch.rand(1, patch_size, patch_size)

    # 将空间维度的随机掩码张量中大于 mask_ratio 的元素置为 1，小于等于 mask_ratio 的元素置为 0
    random_mask_spatial = torch.where(random_mask_spatial > mask_ratio, torch.tensor(1.0), torch.tensor(0.0))

    # 将输入图像的每个 patch 区域与随机掩码逐元素相乘
    masked_image = input_image * random_mask_channels * random_mask_spatial

    # 返回掩码后的图像
    return masked_image

# batch image random mask
def random_mask_batch_image(input_batch, mask_ratio): # input (batchsize, 128, 7, 7)
    batch_size = input_batch.shape[0]
    num_channels = input_batch.shape[1]
    patch_size = input_batch.shape[2]

    # 生成通道维度的随机掩码张量
    # random_mask_channels = torch.rand(batch_size, num_channels, 1, 1)

    # 将通道维度的随机掩码张量中大于 mask_ratio 的元素置为 1，小于等于 mask_ratio 的元素置为 0
    # random_mask_channels = torch.where(random_mask_channels > mask_ratio, torch.tensor(1.0), torch.tensor(0.0))

    # 生成空间维度的随机掩码张量
    random_mask_spatial = torch.rand(batch_size, 1, patch_size, patch_size)

    # 将空间维度的随机掩码张量中大于 mask_ratio 的元素置为 1，小于等于 mask_ratio 的元素置为 0
    random_mask_spatial = torch.where(random_mask_spatial > mask_ratio, torch.tensor(1.0), torch.tensor(0.0))

    # 将输入图像的每个 patch 区域与随机掩码逐元素相乘
    # masked_batch = input_batch * random_mask_channels * random_mask_spatial
    masked_batch = input_batch * random_mask_spatial

    # 返回掩码后的图像 (batchsize, 128, 7, 7)
    return masked_batch

def random_mask_batch_image_mustMaskCenter(input_batch, mask_ratio): # input (batchsize, 128, 7, 7)
    batch_size = input_batch.shape[0]
    num_channels = input_batch.shape[1]
    patch_size = input_batch.shape[2]

    # 生成通道维度的随机掩码张量
    # random_mask_channels = torch.rand(batch_size, num_channels, 1, 1)

    # 将通道维度的随机掩码张量中大于 mask_ratio 的元素置为 1，小于等于 mask_ratio 的元素置为 0
    # random_mask_channels = torch.where(random_mask_channels > mask_ratio, torch.tensor(1.0), torch.tensor(0.0))

    # 生成空间维度的随机掩码张量
    random_mask_spatial = torch.rand(batch_size, 1, patch_size, patch_size)

    # 将空间维度的随机掩码张量中大于 mask_ratio 的元素置为 1，小于等于 mask_ratio 的元素置为 0
    random_mask_spatial = torch.where(random_mask_spatial > mask_ratio, torch.tensor(1.0), torch.tensor(0.0))

    random_mask_spatial[:, :, patch_size // 2, patch_size // 2] = 0

    # 将输入图像的每个 patch 区域与随机掩码逐元素相乘
    # masked_batch = input_batch * random_mask_channels * random_mask_spatial
    masked_batch = input_batch * random_mask_spatial

    # 返回掩码后的图像 (batchsize, 128, 7, 7)
    return masked_batch

def random_mask_batch_image_mustMaskCenter33(input_batch, mask_ratio): # input (batchsize, 128, 7, 7)
    batch_size = input_batch.shape[0]
    num_channels = input_batch.shape[1]
    patch_size = input_batch.shape[2]

    # 生成通道维度的随机掩码张量
    # random_mask_channels = torch.rand(batch_size, num_channels, 1, 1)

    # 将通道维度的随机掩码张量中大于 mask_ratio 的元素置为 1，小于等于 mask_ratio 的元素置为 0
    # random_mask_channels = torch.where(random_mask_channels > mask_ratio, torch.tensor(1.0), torch.tensor(0.0))

    # 生成空间维度的随机掩码张量
    random_mask_spatial = torch.rand(batch_size, 1, patch_size, patch_size)

    # 将空间维度的随机掩码张量中大于 mask_ratio 的元素置为 1，小于等于 mask_ratio 的元素置为 0
    random_mask_spatial = torch.where(random_mask_spatial > mask_ratio, torch.tensor(1.0), torch.tensor(0.0))

    random_mask_spatial[:, :, patch_size // 2 - 1 : patch_size // 2 + 2, patch_size // 2 - 1 : patch_size // 2 + 2] = 0

    # 将输入图像的每个 patch 区域与随机掩码逐元素相乘
    # masked_batch = input_batch * random_mask_channels * random_mask_spatial
    masked_batch = input_batch * random_mask_spatial

    # 返回掩码后的图像 (batchsize, 128, 7, 7)
    return masked_batch


# 输入数据形式 : (patch_size, patch_size, channels) numpy
def Crop_and_resize_single(data, HalfWidth):
    da = transforms.RandomResizedCrop(2 * HalfWidth + 1, scale = (0.08, 1.0), ratio=(0.75, 1.3333333333333333))
    data = data.transpose(2, 0, 1)
    x = da(torch.from_numpy(data))
    x = x.numpy()
    x = x.transpose(1, 2, 0)
    return x