# -*- encoding: utf-8
import numpy as np
import struct
import matplotlib.pyplot as plt

# 图像文件
images_idx3_ubyte_file = 'F:/MNIST/raw/t10k-images.idx3-ubyte'
# 标签文件
labels_idx1_ubyte_file = 'F:/MNIST/raw/t10k-labels.idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_images(idx_ubyte_file=images_idx3_ubyte_file):
    """
    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_labels(idx_ubyte_file=labels_idx1_ubyte_file):
    """
    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)


def load_data():
    images = load_images()
    labels = load_labels()

    # 查看前十个数据及其标签以读取是否正确
    # for i in range(10):
    #     print(train_labels[i])
    #     plt.imshow(train_images[i], cmap='gray')
    #     plt.show()
    # print('done')
    # for i in range(len(images)):
    #     image_mean = np.mean(images[i])
    #     image_stdd = np.std(images[i])
    #     images[i] = (images[i] - image_mean) / image_stdd

    images = images.astype(np.float32)
    labels = labels.astype(np.int64)

    div = int(images.shape[0] * 0.8)
    return images[: div], labels[: div], images[div:], labels[div:]


if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = load_data()
    print(train_images[0])

