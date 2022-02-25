"""
IMDB-Sentiment-Classification  # Code modification by WangYH-BUPT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import ImdbDataset, collate_fn_network
from build_vocab import My_vocab
from lib import sequence_max_len, device, embedding_dim, input_size, hidden_size, train_batch_size, test_batch_size, num_layers, dropout


voc_model = pickle.load(open("./models/vocab.pkl", "rb"))


def get_dataloader(train=True):
    imdb_dataset = ImdbDataset(train)
    batch_size = train_batch_size if train else test_batch_size
    my_dataloader = DataLoader(imdb_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_network)
    return my_dataloader


class ImdbModel(nn.Module):
    def __init__(self):
        super(ImdbModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(voc_model), embedding_dim=embedding_dim, padding_idx=voc_model.PAD)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.fc1 = nn.Linear(64*2, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, input):
        """
        :param input: [batch_size, sequence_max_len]
        :return:
        """
        input_embedded = self.embedding(input)  # input embedded: [batch_size, sequence_max_len, embedding_dim]
        output, (h_n, c_n) = self.lstm(input_embedded)  # h_n: [4, batch_size, hidden_size]
        out = torch.cat([h_n[-1, :, :], h_n[-2, :, :]], dim=-1)  # 拼接正向最后一个输出和反向最后一个输出
        # out: [batch_size, hidden_size*2]
        out_fc1 = self.fc1(out)
        out_fc1_relu = F.relu(out_fc1)
        out_fc2 = self.fc2(out_fc1_relu)  # out: [batch_size, 2]
        return F.log_softmax(out_fc2, dim=-1)


def train(imdb_model, epoch):
    """
    :param imdb_model:
    :param epoch:
    :return:
    """
    train_dataloader = get_dataloader(train=True)

    optimizer = Adam(imdb_model.parameters())
    for i in range(epoch):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        for idx, (data, target) in enumerate(bar):
            optimizer.zero_grad()
            data = data.to(device())
            target = target.to(device())
            output = imdb_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            bar.set_description("epoch:{}  idx:{}   loss:{:.6f}".format(i, idx, loss.item()))


# 验证
def test(imdb_model):
    """
    :param imdb_model:
    :return:
    """
    test_loss = 0
    correct = 0
    imdb_model.eval()
    test_dataloader = get_dataloader(train=False)
    with torch.no_grad():
        for data, target in tqdm(test_dataloader):
            data = data.to(device())
            target = target.to(device())
            output = imdb_model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]  # 获取最大值的位置,[batch_size,1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_dataloader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)))


if __name__ == '__main__':
    imdb_model = ImdbModel().to(device())
    train(imdb_model, 6)
    test(imdb_model)
