from keras.layers import Embedding, Dense, Input, Flatten, Bidirectional, GRU, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.merge import Concatenate
from keras.models import Model


class Models:
    def __init__(self,
                 num_class,
                 num_words,
                 sequence_length_content,
                 sequence_length_summary,
                 embedding_size,
                 embedding_matrix):
        self.num_class = num_class
        self.num_words = num_words
        self.max_seq_len_1 = sequence_length_content
        self.max_seq_len_2 = sequence_length_summary
        self.embedding_size = embedding_size
        self.embedding_matrix = embedding_matrix

        self.text_cnn_extractor = None

    def text_cnn(self,
                 num_filters,
                 filter_sizes):
        input = Input(shape=(self.max_seq_len_2,), dtype='int32')
        embedding_layer = Embedding(input_dim=self.num_words,
                                    output_dim=self.embedding_size,
                                    weights=[self.embedding_matrix],
                                    input_length=self.max_seq_len_2)(input)  # b * max_seq_len * embedding_size

        # create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for filter_size in filter_sizes:
            x = Conv1D(filters=num_filters, kernel_size=filter_size, activation='relu')(
                embedding_layer)  # b * max_seq_len * num_filters
            x = MaxPooling1D(int(x.shape[1]))(x)  # b * num_filters
            pooled_outputs.append(x)

        merged = Concatenate()(pooled_outputs) if len(pooled_outputs) > 1 else pooled_outputs[0]
        x = Flatten(name='text_cnn_feature')(merged)

        outputs = Dense(self.num_class, activation='softmax')(x)

        model = Model(input, outputs)
        print(model.summary())

        return model

    # def get_text_cnn(self,
    #                  num_filters,
    #                  filter_sizes):
    #     self.text_cnn(num_filters, filter_sizes)
    #
    #     return self.text_cnn_extractor

    def multi_channel_model(self,
                            num_filters,
                            filter_sizes,
                            hidden_size):
        input = Input(shape=(self.max_seq_len_1,), dtype='int32')
        embedding_layer = Embedding(input_dim=self.num_words,
                                    output_dim=self.embedding_size,
                                    weights=[self.embedding_matrix],
                                    input_length=self.max_seq_len_1)(input)
        bigru = Bidirectional(GRU(hidden_size, return_state=False, return_sequences=False),
                              merge_mode='concat')(embedding_layer)  # b * (2 * hidden_size)
        x = Dense(32, activation='relu')(bigru)
        feature_1 = Dense(16, activation='relu')(x)

        text_cnn_model = self.text_cnn(num_filters, filter_sizes)
        feature_2 = text_cnn_model.get_layer(name='text_cnn_feature').output
        feature_2 = Dense(16, activation='relu')(feature_2)

        x = Concatenate()([feature_1, feature_2])  # b * (2 * 16)

        dropout = Dropout(0.5, seed=123)(x)  # b * (2 * 16)

        output = Dense(self.num_class, activation='softmax')(dropout)

        model = Model(inputs=[input, text_cnn_model.input], outputs=output)
        print(model.summary())

        return model
