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
# from .modelutils import Trainer, StopTrainingCb
from modelutils import Trainer, StopTrainingCb

def setup():
  cmd = "wget --no-check-certificate https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
  os.system(cmd)
  cmd = 'wget --no-check-certificate "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip" -O "/tmp/cats-and-dogs.zip"'
  os.system(cmd)
  local_zip = '/tmp/cats-and-dogs.zip'
  zip_ref = zipfile.ZipFile(local_zip, 'r')
  zip_ref.extractall('/tmp')
  zip_ref.close()
# UTILS END


class DogVsCat(Trainer):
    def __init__(self, input_shape, n_classes):
        super(DogVsCat, self).__init__(input_shape, n_classes)
        self.model_arch()

    def model_arch(self):
        return self.model_arch_v3()

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
        outputs = keras.layers.Dense(1, activation="sigmoid")(x)
        self.model = keras.Model(inputs, outputs)

    def plot_history(self):
        pio.renderers.default = "colab"
        history_df = pd.DataFrame(self.history.history)
        fig = px.line(history_df)
        fig.show()


def get_y_labels(x):
    return Path(x).parent.name
    # return Path(x).parent.name.lower() == 'Dog'.lower()


def main():
    train_size, valid_size = [.8, .2]
    base_dir_path = "/tmp/cats_and_dogs_filtered"

    train_dir_path = base_dir_path + "/train"
    valid_dir_path = base_dir_path + "/validation"

    # path = Path(base_train_dir_path)
    # fnames = list(path.rglob("*.jpg"))
    # dataset = pd.DataFrame({"fnames": fnames})
    # dataset['filename'] = dataset.fnames.apply(lambda x: "{}/{}".format(Path(x).parent.name, Path(x).name))
    # dataset['y_col'] = dataset.fnames.apply(get_y_labels)
    # idx = list(dataset.index)
    # np.random.seed(42)
    # np.random.shuffle(idx)
    # dataset = dataset.reindex(idx).copy()
    # dataset['is_valid'] = 0
    # dataset['is_valid'].iloc[slice(int(len(dataset) * train_size), len(dataset))] = 1

    train_gen = image.ImageDataGenerator(rescale=1 / 255.,
                                         rotation_range=40,
                                         width_shift_range=.2,
                                         height_shift_range=.2,
                                         shear_range=.2,
                                         zoom_range=.2,
                                         horizontal_flip=True)

    train_gen_obj = train_gen.flow_from_directory(directory=train_dir_path,
                                                  target_size=(150, 150),
                                                  class_mode="binary",
                                                  batch_size=20
                                                  )

    valid_gen = image.ImageDataGenerator(rescale=1 / 255.)
    valid_gen_obj = train_gen.flow_from_directory(directory=valid_dir_path,
                                                  target_size=(150, 150),
                                                  class_mode="binary",
                                                  batch_size=20
                                                  )

    input_shape = (150, 150, 3)
    l = DogVsCat(input_shape=input_shape, n_classes=1)
    l.register_callbacks(StopTrainingCb())
    l.compile()
    l.fit_generator(train_gen_obj,
                    epochs=20,
                    steps_per_epoch=100,
                    valid_data_gen=valid_gen_obj,
                    validation_steps=50)
    l.plot_history()


# MAIN
if __name__ == '__main__':
    # once
    # setup()
    main()
