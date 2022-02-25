# IMDB-Sentiment-Classification

利用Bi-LSTM进行IMDB数据集的情感分类。

### Requirements
* python 3
* torch = 1.7.1
* tqdm


### 准备 vocab.pkl

* 运行 `build_vocab.py` 构建 `vocab.pkl`

### 参数设置

* `lib.py` 存放了参数的设置

### Training & Testing

	python imdb_lstm_model.py
