import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib as plt
from wordcloud import WordCloud, STOPWORDS

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from data_cleaner import DataCleaner

_file_path = os.path.dirname(os.path.realpath(__file__))
EMBEDDING_FILE = _file_path + '/embeddings/wiki-news-300d-1M.vec'
DATA_FILE = _file_path + '/data/Musical_instruments_reviews.csv'

MAX_WORDS_NUM = 50000
MAX_SEQ_LEN_CONTENT = 500
MAX_SEQ_LEN_SUMMARY = 50
EMBEDDING_SIZE = 300
NUM_CLASSES = 5


class DataProcess:
    def __init__(self):
        self.data_cleaner = DataCleaner()

        self.embeddings_index = {}
        self.tokenizer = None
        self.embedding_matrix = None
        self.token_num = None

        self.test_raw_label = None
        self.train_x, self.train_y, self.test_x, self.test_y = None, None, None, None

        data, labels = self.read_data()
        self.to_sequence(data, labels)

        word_to_embedding = self.load_embedding()
        self.calc_embedding_matrix(word_to_embedding)

    def read_data(self):
        data = pd.read_csv(DATA_FILE)

        # check na value first
        data.isna().sum()

        # del the na row
        data = data.dropna(axis=0, subset=['reviewText'])

        # data['text'] = data['reviewText'] + ' ' + data['summary']
        # data['text'] = data['text'].astype('str').apply(self.data_cleaner.denoise_text)
        data['reviewText'] = data['reviewText'].apply(self.data_cleaner.denoise_text)
        data['summary'] = data['summary'].apply(self.data_cleaner.denoise_text)

        data['label'] = data['overall'].map(lambda x: x - 1)
        labels = to_categorical(data['label'], num_classes=5)

        # calc some col's mean of length to decide the MAX_SEQ_LEN
        s = data.summary.str.len()
        print(s.describe())

        # look up date freq distribution
        data.overall.value_counts()

        # wordcloud visualize
        """
        plt.figure(figsize=(20, 20))
        # fake
        wc = WordCloud(max_words=2000, width=1600, height=800, stopwords=STOPWORDS).generate(
            ' '.join(data[data.label == 1].text))
        plt.imshow(wc, interpolation='bilinear')
        plt.savefig('fake_wc.png')
        # real
        wc = WordCloud(max_words=2000, width=1600, height=800, stopwords=STOPWORDS).generate(
            ' '.join(data[data.label == 0].text))
        plt.imshow(wc, interpolation='bilinear')
        plt.savefig('real_wc.png')
        """

        def count(data, words_to_cnt):
            for sample in data:
                for k, v in sample.items():
                    if k == 'title' or k == 'text':
                        v = v.strip().split(' ')

                        for word in v:
                            words_to_cnt[word] = words_to_cnt.get(word, 0) + 1

            words_to_cnt = sorted(words_to_cnt.items(), key=lambda x: x[1], reverse=True)
            print(words_to_cnt[50000])

        # count(fake_news, {})
        # count(true_news, {})

        print("data count: ", len(data))

        return data, labels

    def load_embedding(self):
        word_to_embedding = {}

        with open(EMBEDDING_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split(" ")

                if len(line) != 301:
                    continue

                word = line[0]
                coefs = np.asarray(line[1:], dtype='float32')

                word_to_embedding[word] = coefs

        print('embedding count: ', len(word_to_embedding))
        return word_to_embedding

    def to_sequence(self, data, labels):
        train_content, test_content, self.train_y, self.test_y = train_test_split(data.reviewText, labels,
                                                                                  test_size=0.2,
                                                                                  random_state=123)
        train_summary, test_summary, self.train_y, self.test_y = train_test_split(data.summary, labels,
                                                                                  test_size=0.2,
                                                                                  random_state=123)
        _, _, _, self.test_raw_label = train_test_split(data.summary, data.label, test_size=0.2, random_state=123)

        self.tokenizer = Tokenizer(num_words=MAX_WORDS_NUM)
        self.tokenizer.fit_on_texts(train_content + train_summary)

        tokenizer_train_1 = self.tokenizer.texts_to_sequences(train_content)
        train_content = pad_sequences(tokenizer_train_1, maxlen=MAX_SEQ_LEN_CONTENT)

        tokenizer_train_2 = self.tokenizer.texts_to_sequences(train_summary)
        train_summary = pad_sequences(tokenizer_train_2, maxlen=MAX_SEQ_LEN_SUMMARY)

        tokenizer_test_1 = self.tokenizer.texts_to_sequences(test_content)
        test_content = pad_sequences(tokenizer_test_1, maxlen=MAX_SEQ_LEN_CONTENT)

        tokenizer_test_2 = self.tokenizer.texts_to_sequences(test_summary)
        test_summary = pad_sequences(tokenizer_test_2, maxlen=MAX_SEQ_LEN_SUMMARY)

        self.train_x, self.test_x = (train_content, train_summary), (test_content, test_summary)

    def calc_embedding_matrix(self, word_to_embedding):
        all_embs = np.stack(word_to_embedding.values())
        emb_mean, emb_std = all_embs.mean(), all_embs.std()

        word_to_index = self.tokenizer.word_index
        self.token_num = min(MAX_WORDS_NUM, len(word_to_index)) + 1
        self.embedding_matrix = np.random.normal(emb_mean, emb_std, (self.token_num, EMBEDDING_SIZE))

        for word, idx in word_to_index.items():
            if idx >= MAX_WORDS_NUM:
                continue

            embedding_vector = word_to_embedding.get(word, None)
            if embedding_vector is not None:
                self.embedding_matrix[idx] = embedding_vector

        print('embedding matrix cnt: ', len(self.embedding_matrix))

    def get_token_nums(self):
        return self.token_num

    def get_embedding_matrix(self):
        return self.embedding_matrix

    def get_data(self):
        return self.train_x, self.test_x, self.train_y, self.test_y

    def get_raw_label(self):
        return self.test_raw_label
