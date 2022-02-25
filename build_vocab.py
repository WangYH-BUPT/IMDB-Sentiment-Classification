"""
文本序列化
"""

import pickle
from tqdm import tqdm

from dataset import ImdbDataset, collate_fn_vocab
from torch.utils.data import DataLoader


class My_vocab:
    UNK_TAG = "<UNK>"  # 表示未知字符
    PAD_TAG = "<PAD>"  # 填充符
    PAD = 0
    UNK = 1

    def __init__(self):
        self.dict = {  # 保存词语和对应的数字
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }
        self.count = {}  # 统计词频

    def fit(self, sentence):
        """
        接受句子，统计词频
        :param sentence:[str,str,str]
        :return:None
        """
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1  # 所有的句子fit之后，self.count就有了所有词语的词频

    def build_vocab(self, min_count=1, max_count=None, max_features=None):
        """
        根据条件构造 词典
        :param min_count:最小词频
        :param max_count: 最大词频
        :param max_features: 最大词语数
        :return:
        """
        # 删除count中词频小于min_count的word
        if min_count is not None:
            self.count = {word: count for word, count in self.count.items() if count >= min_count}
        # 删除count中词频大于max的word
        if max_count is not None:
            self.count = {word: count for word, count in self.count.items() if count <= max_count}
        # 限制保留的词语数
        if max_features is not None:
            # [(k,v),(k,v)....] --->{k:v,k:v}
            temp = sorted(self.count.items(), key=lambda x: x[-1], reverse=True)[:max_features]
            self.count = dict(temp)
        # 每次word对应一个数字
        for word in self.count:
            self.dict[word] = len(self.dict)

        # 把dict进行翻转
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence, max_len=None):
        """
        把句子转化为数字序列
        :param sentence:[str,str,str]
        :return: [int,int,int]
        """
        if len(sentence) > max_len:
            sentence = sentence[:max_len]
        else:
            sentence = sentence + [self.PAD_TAG] * (max_len - len(sentence))  # 填充PAD

        return [self.dict.get(i, 1) for i in sentence]

    def inverse_transform(self, incides):
        """
        把数字序列转化为字符
        :param incides: [int,int,int]
        :return: [str,str,str]
        """
        return [self.inverse_dict.get(i, "<UNK>") for i in incides]

    def __len__(self):
        return len(self.dict)


def mydataloader(train=True):
    imdb_dataset = ImdbDataset(train)
    my_dataloader = DataLoader(imdb_dataset, batch_size=200, shuffle=True, collate_fn=collate_fn_vocab)
    return my_dataloader


if __name__ == '__main__':
    word2seq = My_vocab()
    dl_train = mydataloader(True)
    dl_test = mydataloader(False)
    for reviews, label in tqdm(dl_train, total=len(dl_train)):
        for sentence in reviews:
            word2seq.fit(sentence)
    for reviews, label in tqdm(dl_test, total=len(dl_test)):
        for sentence in reviews:
            word2seq.fit(sentence)
    word2seq.build_vocab()
    print(len(word2seq))
    pickle.dump(word2seq, open("./models/vocab.pkl", "wb"))
