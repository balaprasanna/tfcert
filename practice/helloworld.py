import tensorflow as tf
from tensorflow import keras
import numpy as np


class M(object):

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
    print(tf.__version__)
    xs = np.array([-1.0, 0.0, 1.0, 2.0], dtype=float)
    ys = (xs * 3) - 2
    print(ys)
    model = M()
    model.compile()
    model.fit(xs, ys, epochs=200)
    model.predict([10.0] , (10. * 3) - 2)
    model.fit(xs, ys, epochs=200)
    model.predict([10.0] , (10. * 3) - 2)


if __name__ == '__main__':
    main()
