import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.io as pio
import plotly.express as px
import zipfile
import os

# UTILS BEGIN
#from .modelutils import Trainer, StopTrainingCb
from modelutils import Trainer, StopTrainingCb

def setup():
  cmd = 'wget --header="Host: storage.googleapis.com" --header="User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.92 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9" --header="Accept-Language: en-US,en;q=0.9,ta;q=0.8" --header="Referer: https://www.kaggle.com/" "https://storage.googleapis.com/kaggle-data-sets/3258%2F5337%2Fbundle%2Farchive.zip?GoogleAccessId=gcp-kaggle-com@kaggle-161607.iam.gserviceaccount.com&Expires=1594203433&Signature=KcwWJiFGq5HuK4Q8PkWmaYQZm%2BPmwgR0n5nKYksMjorwDyks1twWZNnYPjle62ae4yxsnHS4Fctn4GaVTm5tQuieHbdjPkJx23eIve0fdZIbIJKgYcvDpFSFnLnjgCAd4XtdD%2FxSsFcswrUrchz48eFEGoGQD3SL7aGVLAcVGbVPSKJrlVI3qhf9qTvXKxQdb0rDz3OLnGcgDJ7hN%2Bndm%2BfMWgp6%2FP9nIomBDKAffxYQHrvwuVmTVdgc69sX3tbNJj1AHqNU4azT5Kgv4NcQL5AvUZIhqZBwOhjMo87hRotwgpini0HBAA5AJPLKbK%2BuXAgAAPUssHWlkr4AiZ6lrg%3D%3D" -c -O "/tmp/sign.zip"'
  os.system(cmd)
  local_zip = '/tmp/sign.zip'
  zip_ref = zipfile.ZipFile(local_zip, 'r')
  zip_ref.extractall('/tmp')
  zip_ref.close()
# UTILS END


class SignLanguage(Trainer):
    def __init__(self, input_shape, n_classes):
        super(SignLanguage, self).__init__(input_shape, n_classes)

    # def model_arch(self):
    #     return self.model_arch_v3()

    def model_arch_v2(self):
        base_model = tf.keras.applications.inception_v3.InceptionV3(
            input_shape=self.input_shape,
            include_top=False,
            weights=None)
        weights_file = "/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
        base_model.load_weights(weights_file)

        for layer in base_model.layers:
            layer.trainable = False

        last_layer = base_model.get_layer("mixed7")
        last_output = last_layer.output

        x = tf.keras.layers.Flatten()(last_output)
        x = tf.keras.layers.Dense(1024, activation="relu")(x)
        x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(base_model.input, x)
        self.model = model
        return model

    def model_arch_v3(self):
        base_model = tf.keras.applications.inception_v3.InceptionV3(
            input_shape=self.input_shape,
            include_top=False,
            weights=None)
        weights_file = "/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
        base_model.load_weights(weights_file)
        base_model.trainable = False
        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(self.n_classes, activation="sigmoid")(x)
        self.model = keras.Model(inputs, outputs)

    def plot_history(self):
        pio.renderers.default = "browser" #colab
        history_df = pd.DataFrame(self.history.history)
        print(history_df)
        fig = px.line(history_df)
        fig.show()


def get_y_labels(x):
    return Path(x).parent.name
    # return Path(x).parent.name.lower() == 'Dog'.lower()


def main():
    train_size, valid_size = [.8, .2]

    train_path = "/tmp/sign_mnist_train.csv"
    test_path = "/tmp/sign_mnist_test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_x = train_df.iloc[:, 1:].to_numpy().reshape(-1, 28, 28) #.reshape(-1, 28, 28, 1)
    train_x = np.expand_dims(train_x, axis=3)
    train_y = train_df.iloc[:, 0].to_numpy()

    test_x = test_df.iloc[:, 1:].to_numpy().reshape(-1, 28, 28) #.reshape(-1, 28, 28, 1)
    test_x = np.expand_dims(test_x, axis=3)
    test_y = test_df.iloc[:, 0].to_numpy()

    print("train", train_x.shape)
    print("valid", test_x.shape)
    train_gen = image.ImageDataGenerator(rescale=1 / 255.,
                                         rotation_range=40,
                                         width_shift_range=.2,
                                         height_shift_range=.2,
                                         shear_range=.2,
                                         zoom_range=.2,
                                         horizontal_flip=True)

    train_gen_obj = train_gen.flow(train_x, train_y, batch_size=32)
    valid_gen = image.ImageDataGenerator(rescale=1 / 255.)
    valid_gen_obj = valid_gen.flow(test_x, test_y, batch_size=32)

    input_shape = (28, 28, 1)
    l = SignLanguage(input_shape=input_shape, n_classes=26)
    l.register_callbacks(StopTrainingCb())
    l.compile()
    l.model.summary()
    l.fit_generator(train_gen_obj,
                    epochs=1,
                    steps_per_epoch=857,
                    valid_data_gen=valid_gen_obj,
                    validation_steps=224)
    l.plot_history()


# MAIN
if __name__ == '__main__':
    # once
    # setup()
    main()

