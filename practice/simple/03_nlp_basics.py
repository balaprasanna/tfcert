import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import pandas as pd


def setup():
    cmd = "wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json -O /tmp/sarcasm.json"
    os.system(cmd)


class BaseTextModel(object):
    def __init__(self):
        self.model = self.get_arch()

    def load_dataset(self):
        df = pd.read_json(Params.input_fname)
        df = df[['headline', 'is_sarcastic']]

        self.train_df = df.sample(frac=.8, replace=False, random_state=42)
        self.test_df = df.drop(index=self.train_df.index)
        print(self.train_df.shape)
        print(self.test_df.shape)

    def tokenize_and_padding(self):
        self.train_sentences = self.train_df.headline.values.tolist()
        self.train_labels = self.train_df.is_sarcastic.values

        self.test_sentences = self.test_df.headline.values.tolist()
        self.test_labels = self.test_df.is_sarcastic.values

        self.tokenizer = Tokenizer(num_words=Params.vocab_size, oov_token=Params.oov_token)
        self.tokenizer.fit_on_texts(self.train_sentences)
        self.train_word_index = self.tokenizer.word_index

        self.train_seqs = self.tokenizer.texts_to_sequences(self.train_sentences)
        self.train_seqs_padded = pad_sequences(self.train_seqs,
                                          maxlen=Params.maxlen,
                                          padding=Params.padding_mode,
                                          truncating=Params.truncate_type)

        self.test_seqs = self.tokenizer.texts_to_sequences(self.test_sentences)
        self.test_seqs_padded = pad_sequences(self.test_seqs,
                                         maxlen=Params.maxlen,
                                         padding=Params.padding_mode,
                                         truncating=Params.truncate_type)

    def get_arch(self):
        return tf.keras.Sequential([
            Embedding(input_dim=Params.vocab_size,
                      output_dim=Params.emb_dim,
                      input_length=Params.maxlen),
            GlobalAveragePooling1D(),
            Dense(24, 'relu'),
            Dense(1, 'sigmoid')
        ])

    def compile(self):
        self.model.compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.losses.binary_crossentropy,
            metrics=['acc']
        )

    def fit(self):
        self.history = self.model.fit(
            x=self.train_seqs_padded, y=self.train_labels,
            batch_size=32, epochs=10, verbose=2,
            validation_data=(self.test_seqs_padded, self.test_labels)
        )

    def plot_history(self, renderer="browser"):
        import plotly.express as pe
        history_df = pd.DataFrame(self.history.history)
        print(history_df)
        fig1 = pe.line(history_df[['acc', 'val_acc']])
        fig2 = pe.line(history_df[['loss', 'val_loss']])
        fig1.show(renderer)
        fig2.show(renderer)

    def parse_embeddings(self):
        # export the embeddings
        e = self.model.layers[0]
        weights = e.get_weights()[0]
        print(weights.shape)  # shape: (vocab_size, embedding_dim)


class Params:
    input_fname = "/tmp/sarcasm.json"
    vocab_size = 10000
    emb_dim = 16
    oov_token = '<oov>'
    maxlen = 100
    padding_mode = 'post'
    truncate_type = 'post'


def main():
    m = BaseTextModel()
    m.load_dataset()
    m.tokenize_and_padding()
    m.compile()
    m.fit()
    m.plot_history()
    m.parse_embeddings()

    train_index_word = m.tokenizer.index_word
    for i in range(2):
        original = m.train_sentences[i]
        transformed = [train_index_word.get(idx, 'pad') for idx in m.train_seqs_padded[i]]
        print("org: ", original)
        print("new: ", " ".join(transformed))


if __name__ == '__main__':
    # setup()
    main()
