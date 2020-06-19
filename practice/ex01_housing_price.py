import tensorflow as tf
from tensorflow import keras
import numpy as np


class HousingPriceModel(object):

    def __init__(self):
        self.model = self.arch()
        self.optimizer = keras.optimizers.SGD()
        self.loss = keras.losses.mean_squared_error

    def arch(self):
        return tf.keras.Sequential(
            layers=[
                keras.layers.Dense(units=1, input_shape=[1])
            ]
        )

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def fit(self, x, y, epochs=500):
        self.model.fit(x, y, epochs=epochs)

    def predict(self, x, y=None):
        yhat = self.model.predict(x)
        if y:
            print(f"Predicted {yhat} vs Actual {y}")
        else:
            print(f"Predicted {yhat}")


def main(*args):

    xs = np.array([0, 1, 2, 3, 20], dtype=float)

    def gen_y_func(x):
        base = 50
        no_of_bedrooms = x
        y = base + (no_of_bedrooms * base)
        return y

    ys = np.array([gen_y_func(x) for x in xs], dtype=float)
    print(xs)
    print(ys)

    model = HousingPriceModel()
    model.compile()
    model.fit(xs, ys, epochs=200)
    model.predict([7], gen_y_func(7))



if __name__ == '__main__':
    main()
