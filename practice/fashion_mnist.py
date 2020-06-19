import tensorflow as tf
from tensorflow import keras
import numpy as np


class Dataset(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def show_batch(self, bs=1):
        pass


class FashionMnistModel(object):

    def __init__(self):
        self.model = self.arch()
        self.optimizer = keras.optimizers.Adam()
        self.loss = keras.losses.sparse_categorical_crossentropy

    def arch(self):
        return tf.keras.Sequential(
            layers=[
                keras.layers.Flatten(input_shape=(28, 28)),
                keras.layers.Dense(units=128, activation=tf.nn.relu),
                keras.layers.Dense(units=10, activation=tf.nn.softmax)
            ]
        )

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def fit(self, x, y, epochs=500):
        self.model.fit(x, y, epochs=epochs)

    def predict(self, x, y=None):
        yhat = self.model.predict(x)
        if y:
            print(f"Predicted {yhat.argmax()} vs Actual {y}")
        else:
            print(f"Predicted {yhat}")



def main(*args):

    fashion_mnist = keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train/255.0
    x_test = x_test / 255.0

    model = FashionMnistModel()
    model.compile()
    model.fit(x_train, y_train, epochs=10)
    model.predict(x_test[6:7], y_test[6:7])
    print(f"Acc:  {model.model.evaluate(x_test, y_test)}")



if __name__ == '__main__':
    main()
