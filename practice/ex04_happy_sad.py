import tensorflow as tf
from tensorflow import keras
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import io
import numpy as np

class StopTrainingCb(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):
        print(logs)
        acc = logs.get('acc')
        if acc > 0.99:
            print(f"Stop training since its reaches 99% Acc: => {acc}")
            self.model.stop_training = True


class HappySadModel(object):

    def __init__(self):
        self.model = self.model_arch()
        self.optimizer = keras.optimizers.Adam()
        self.loss = keras.losses.binary_crossentropy
        self.metrics = [
            "acc",
            # keras.metrics.binary_accuracy
        ]
        # interal stuff
        self.cbs = None

    def model_arch(self):
        return tf.keras.Sequential(
            layers=[
                keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu, input_shape=(128, 128, 3)),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.nn.relu),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dense(units=1024, activation=tf.nn.relu),
                keras.layers.Dense(units=64, activation=tf.nn.relu),
                keras.layers.Dense(units=1, activation=tf.nn.sigmoid)
            ]
        )

    def register_callbacks(self, cbs):
        if not isinstance(cbs, list):
            cbs = [cbs]
        self.cbs = cbs

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def fit(self, x, y, epochs=10):
        if self.cbs:
            self.model.fit(x, y, epochs=epochs, callbacks=self.cbs)
        else:
            self.model.fit(x, y, epochs=epochs)

    def fit_generator(self, train_gen, epochs=10):
        self.model.fit_generator(train_gen,
                                 steps_per_epoch=2,
                                 epochs=epochs,
                                 callbacks=self.cbs)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def predict(self, x, y=None):
        yhat = self.model.predict(x)
        if y:
            print(f"Predicted {yhat.argmax()} vs Actual {y}")
        else:
            print(f"Predicted {yhat}")
        return yhat

    def predict_in_colab(self):
        from google.colab import files
        uploaded = files.upload()
        for k, v in uploaded.items():
            fname = f"/content/{k}"
            print(fname)
            img = image.load_img(fname, target_size=(128, 128))
            x = image.img_to_array(img)
            x = np.expand_dims(x, 0)
            res = model.model.predict(x)
            print(f"file {k} output {res}")
            # v = uploaded['happy.jpeg']
            # img = Image.open(io.BytesIO(v))
            # out[k]=img


if __name__ == '__main__':
    s = time.time()
    train_data_dir = "/opt/localtmp/dataset/happy_sad"
    train_data_gen = ImageDataGenerator(rescale=1./255)
    train_data_gen = train_data_gen.flow_from_directory(train_data_dir,
                                       target_size=(128, 128),
                                       batch_size=10,
                                       class_mode="binary")
    callback = StopTrainingCb()
    model = HappySadModel()
    model.compile()
    # model.model.summary()
    model.register_callbacks(callback)
    model.fit_generator(train_data_gen, epochs=20)
    # model.predict_in_colab()
    e = time.time()
    print(f"It took {e-s} seconds")
