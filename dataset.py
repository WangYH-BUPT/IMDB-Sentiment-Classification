"""
完成数据集的准备
"""

import os
import re
import zipfile
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from lib import sequence_max_len


# 进行文本分词, 数据预处理: 将一些会干扰数据处理的符号删除
def tokenize(content):
    """
    :param content: str
    :return tokens: [str, str, str, ...]
    """
    filters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>', '\?',
                 '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”', '“']

    # 替换处理: 将content的干扰符号集替换成空格, 分词处理
    content = re.sub("<br />", " ", content)
    content = re.sub("I'm", "I am", content)
    content = re.sub("isn't", "is not", content)
    content = re.sub("|".join(filters), " ", content)
    tokens = [i.strip().lower() for i in content.split(" ") if len(i) > 0]
    return tokens


# 解压缩
def unzip_file(zip_src, dst_dir):
    """
    :param zip_src:
    :param dst_dir:
    :return
    """
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, 'r')
        bar = tqdm(fz.namelist())
        bar.set_description("unzip  " + zip_src)
        for file in bar:
            fz.extract(file, dst_dir)
    else:
        print('This is not zip')


# 定义ImdbDataset
class ImdbDataset(Dataset):
    def __init__(self, train=True):
        super(ImdbDataset, self).__init__()
        # 添加文件路径
        if not os.path.exists("./data/unzip"):
            data_path = r"./data/"
            zip_path = []  # ["./data/train.zip", "./data/test.zip"]
            zip_path += [os.path.join(i) for i in os.listdir(data_path) if i.endswith(".zip")]
            for i in zip_path:
                zip_data_path = data_path + i
                unzip_file(zip_data_path, "./data/unzip")
            # 相当于:
            # unzip_file("./data/test.zip", "./data/unzip")
            # unzip_file("./data/train.zip", "./data/unzip")

        data_path = r"./data/unzip"
        data_path += r"/train" if train else r"/test"
        # 保存所有的文件路径
        self.total_path = []
        for temp_path in [r"/pos", r"/neg"]:
            cur_path = data_path + temp_path
            self.total_path += [os.path.join(cur_path, i) for i in os.listdir(cur_path) if i.endswith(".txt")]

    def __getitem__(self, idx):
        file = self.total_path[idx]
        review = tokenize(open(file, "r", encoding="utf-8").read())  # 从txt获取评论并分词
        label = int(file.split("_")[-1].split(".")[0])  # 获取评论对应的label
        label = 0 if label < 5 else 1
        return review, label

    def __len__(self):
        return len(self.total_path)


# 调试, 对batch数据进行处理, collate_fn能够解决batch_size!=1报错的问题
def collate_fn_vocab(batch):
    """
    :param batch: 里面是一个一个getitem的结果: [[tokens, label], [tokens, label], [tokens, label], ...]
    :return: 元组
    """
    reviews, labels = zip(*batch)
    return reviews, labels


def collate_fn_network(batch):
    """
    :param batch: 里面是一个一个getitem的结果: [[tokens, label], [tokens, label], [tokens, label], ...]
    :return: 元组
    """
    reviews, labels = zip(*batch)
    voc_model = pickle.load(open("./models/vocab.pkl", "rb"))
    reviews = [voc_model.transform(i, max_len=sequence_max_len) for i in reviews]
    reviews, labels = torch.LongTensor(reviews), torch.LongTensor(labels)
    return reviews, labels


# 构建test_file
def test_file(train=True):
    if not os.path.exists("./data/unzip"):
        unzip_file("./data/data.zip", "./data/unzip")
    data_path = r"./data/unzip"
    data_path += r"/train" if train else r"/test"
    total_path = []  # 保存所有的文件路径
    for temp_path in [r"/pos", r"/neg"]:
        cur_path = data_path + temp_path
        total_path += [os.path.join(cur_path, i) for i in os.listdir(cur_path) if i.endswith(".txt")]
    print(total_path)


if __name__ == "__main__":
    imbd_dataset = ImdbDataset(True)
    my_dataloader = DataLoader(imbd_dataset, batch_size=200, shuffle=True, collate_fn=collate_fn_vocab)
    for data in my_dataloader:
        voc_model = pickle.load(open("./models/vocab.pkl", "rb"))
        break
