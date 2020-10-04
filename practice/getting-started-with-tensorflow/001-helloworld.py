#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras.layers import *
import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv('./data/mnist.csv', header=None)
    x_train = df.iloc[:, 1:].values
    y_train = df.iloc[:, 0].values
    x_train = x_train / 255.

    model = tf.keras.Sequential([
        Dense(16, activation="relu"),
        Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=['acc'])
    hist = model.fit(x_train, y_train, epochs=3, batch_size=32)

    pd.DataFrame(hist.history)['loss'].plot(kind="line")
    plt.show()
    pd.DataFrame(hist.history)['acc'].plot(kind="line")
    plt.show()

if __name__ == '__main__':
    main()





