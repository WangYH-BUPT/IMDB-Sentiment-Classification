"""
参数  # Code modification by WangYH-BUPT
"""

import torch


sequence_max_len = 100
train_batch_size = 512
test_batch_size = 128

embedding_dim = 200
input_size = 200
hidden_size = 64
num_layers = 2
dropout = 0.5


def device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
