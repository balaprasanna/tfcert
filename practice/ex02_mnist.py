import tensorflow as tf
from tensorflow import keras
import time


class StopTrainingCb(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):
        print(logs)
        acc = logs.get('sparse_categorical_accuracy')
        if acc > 0.99:
            print(f"Stop training since its reaches 99% Acc: => {acc}")
            self.model.stop_training = True


class MnistModel(object):

    def __init__(self):
        self.model = self.model_arch()
        self.optimizer = keras.optimizers.Adam()
        self.loss = keras.losses.sparse_categorical_crossentropy

        # interal stuff
        self.cbs = None

    def model_arch(self):
        return tf.keras.Sequential(
            layers=[
                keras.layers.Flatten(input_shape=(28, 28)),
                keras.layers.Dense(units=128, activation=tf.nn.relu),
                keras.layers.Dense(units=64, activation=tf.nn.relu),
                keras.layers.Dense(units=10, activation=tf.nn.softmax)
            ]
        )

    def register_callbacks(self, cbs):
        if not isinstance(cbs, list):
            cbs = [cbs]
        self.cbs = cbs

    def compile(self):
        metrics = [tf.keras.metrics.sparse_categorical_accuracy]
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=metrics)

    def fit(self, x, y, epochs=10):
        if self.cbs:
            self.model.fit(x, y, epochs=epochs, callbacks=self.cbs)
        else:
            self.model.fit(x, y, epochs=epochs)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def predict(self, x, y=None):
        yhat = self.model.predict(x)
        if y:
            print(f"Predicted {yhat.argmax()} vs Actual {y}")
        else:
            print(f"Predicted {yhat}")


def main(*args):
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape)

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    callback = StopTrainingCb()

    model = MnistModel()
    model.compile()
    model.register_callbacks(callback)
    model.fit(x_train, y_train, epochs=20)
    print(f"Train Acc:  {model.evaluate(x_train, y_train)}")
    print(f"Test Acc:  {model.evaluate(x_test, y_test)}")


if __name__ == '__main__':
    s = time.time()
    main()
    e = time.time()
    print(f"It took {e-s} seconds")
