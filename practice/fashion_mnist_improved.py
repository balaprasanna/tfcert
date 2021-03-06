import tensorflow as tf
from tensorflow import keras


class MyCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        lr = self.model.optimizer._get_hyper("learning_rate").value()
        self.model.optimizer._set_hyper("learning_rate", lr / 2.)
        newlr = self.model.optimizer._get_hyper("learning_rate").value()
        print(f"lr => {lr} ,  newlr: => {newlr}")

    def on_epoch_end(self, epoch, logs={}):
        l = logs.get('loss')
        print(f"on epoch end: => {l}")
        if l < 0.4:
            print(f"Stoping straining with loss: => {l}")
            self.model.stop_training = True


class FashionMnistV2Model(object):

    def __init__(self):
        self.model = self.arch()
        self.optimizer = keras.optimizers.Adam()
        self.loss = keras.losses.sparse_categorical_crossentropy

    def arch(self):
        return tf.keras.Sequential(
            layers=[
                keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu, input_shape=(28, 28, 1)),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.nn.relu),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dense(units=1024, activation=tf.nn.relu),
                keras.layers.Dense(units=64, activation=tf.nn.relu),
                keras.layers.Dense(units=10, activation=tf.nn.softmax)
            ]
        )

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def fit(self, x, y, epochs=500, cbs=None):
        if cbs:
            self.model.fit(x, y, epochs=epochs, callbacks=cbs)
        else:
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

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    print(x_train.shape)
    print(x_test.shape)
    model = FashionMnistV2Model()
    model.compile()
    cbs = [MyCallback()]
    model.fit(x_train, y_train, epochs=10, cbs=cbs)
    model.predict(x_test[6:7], y_test[6:7])
    print(f"Acc:  {model.model.evaluate(x_test, y_test)}")


if __name__ == '__main__':
    main()
