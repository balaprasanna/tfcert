import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

def predict_in_colab(model):
    from google.colab import files
    uploaded = files.upload()
    for k, v in uploaded.items():
        fname = f"/content/{k}"
        print(fname)
        img = image.load_img(fname, target_size=(128, 128))
        x = image.img_to_array(img)
        x = np.expand_dims(x, 0)
        res = model.predict(x)
        print(f"file {k} output {res}")
        # v = uploaded['happy.jpeg']
        # img = Image.open(io.BytesIO(v))
        # out[k]=img


# =================
#     Base Training Helper
# =================
class Trainer(object):

    def __init__(self, input_shape, n_classes):
        self.input_shape = input_shape  # (128, 128, 3)
        self.n_classes = n_classes
        self.optimizer = keras.optimizers.Adam()

        self.metrics = [
            "acc",
            # keras.metrics.binary_accuracy
        ]
        # interal stuff
        self.cbs = None
        self.final_layer_activation = tf.nn.softmax if self.n_classes > 1 else tf.nn.sigmoid
        self.loss = keras.losses.categorical_crossentropy if self.n_classes > 1 else  keras.losses.binary_crossentropy

        self.model = self.model_arch()

    def model_arch(self):
        return tf.keras.Sequential(
            layers=[
                keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu, input_shape=self.input_shape),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.nn.relu),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dense(units=1024, activation=tf.nn.relu),
                keras.layers.Dense(units=64, activation=tf.nn.relu),
                keras.layers.Dense(units=self.n_classes, activation=self.final_layer_activation)
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
            self.history = self.model.fit(x, y, epochs=epochs, callbacks=self.cbs)
        else:
            self.history = self.model.fit(x, y, epochs=epochs)

    def fit_generator(self, train_gen, valid_data_gen=None,  epochs=10, steps_per_epoch=2, validation_steps=2):
        if valid_data_gen:
            self.history = self.model.fit_generator(train_gen,
                                     steps_per_epoch=steps_per_epoch,
                                     epochs=epochs,
                                     validation_data=valid_data_gen,
                                     validation_steps=validation_steps,
                                     callbacks=self.cbs)
        else:
            self.history = self.model.fit_generator(train_gen,
                                     steps_per_epoch=steps_per_epoch,
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

    def plot_history(self):
        pio.renderers.default = "browser"
        history_df = pd.DataFrame(self.history.history)
        fig = px.line(history_df)
        fig.show()


# =================
#     Callbacks
# =================
class StopTrainingCb(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):
        print(logs)
        train_acc = logs.get('acc')
        if train_acc >= 0.9:
            print(f"Stop training since its reaches 99% Acc: => {train_acc}")
            self.model.stop_training = True

