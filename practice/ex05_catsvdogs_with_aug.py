import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from pathlib import Path

# UTILS BEGIN
# from .modelutils import Trainer, StopTrainingCb
from modelutils import Trainer, StopTrainingCb


# UTILS END

class DogVsCat(Trainer):
    def __init__(self, input_shape, n_classes):
        super(DogVsCat, self).__init__(input_shape, n_classes)
        self.model_arch()

    def model_arch(self):
        return self.model_arch_v2()

    def model_arch_v2(self):
        base_model = tf.keras.applications.inception_v3.InceptionV3(
            input_shape=self.input_shape,
            include_top=False,
            weights=None)
        weights_file = "/home/prasanna/Downloads/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
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


def get_y_labels(x):
    return Path(x).parent.name
    # return Path(x).parent.name.lower() == 'Dog'.lower()


def main():
    train_size, valid_size = [.8, .2]
    base_train_dir_path = "/opt/localtmp/dataset/PetImages"
    path = Path(base_train_dir_path)
    fnames = list(path.rglob("*.jpg"))
    dataset = pd.DataFrame({"fnames": fnames})
    dataset['filename'] = dataset.fnames.apply(lambda x: "{}/{}".format(Path(x).parent.name, Path(x).name))
    dataset['y_col'] = dataset.fnames.apply(get_y_labels)
    idx = list(dataset.index)
    np.random.seed(42)
    np.random.shuffle(idx)
    dataset = dataset.reindex(idx).copy()
    dataset['is_valid'] = 0
    dataset['is_valid'].iloc[slice(int(len(dataset) * train_size), len(dataset))] = 1
    print(dataset.head())
    print(dataset.is_valid.value_counts(True))
    print(dataset.y_col.value_counts(True))
    print("train")
    print(dataset[dataset.is_valid == 0].y_col.value_counts(True))
    print("valid")
    print(dataset[dataset.is_valid == 1].y_col.value_counts(True))

    train_gen = image.ImageDataGenerator(rescale=1 / 255.,
                                         rotation_range=40,
                                         width_shift_range=.2,
                                         height_shift_range=.2,
                                         shear_range=.2,
                                         zoom_range=.2,
                                         vertical_flip=True,
                                         preprocessing_function=tf.keras.applications.inception_v3.preprocess_input)

    train_gen_obj = train_gen.flow_from_dataframe(dataset[dataset.is_valid == 0],
                                                  directory=base_train_dir_path,
                                                  x_col="filename",
                                                  y_col='y_col',
                                                  target_size=(150, 150),
                                                  class_mode="binary",
                                                  batch_size=128,
                                                  has_ext=True
                                                  )

    valid_gen = image.ImageDataGenerator(rescale=1 / 255.,
                                         preprocessing_function=tf.keras.applications.inception_v3.preprocess_input)
    valid_gen_obj = valid_gen.flow_from_dataframe(dataset[dataset.is_valid == 1],
                                                  directory=base_train_dir_path,
                                                  x_col="filename",
                                                  y_col='y_col',
                                                  target_size=(150, 150),
                                                  class_mode="binary",
                                                  batch_size=128,
                                                  has_ext=True
                                                  )

    input_shape = (150, 150, 3)
    l = DogVsCat(input_shape=input_shape, n_classes=1)
    l.register_callbacks(StopTrainingCb())
    l.compile()
    l.fit_generator(train_gen_obj,
                    epochs=10,
                    steps_per_epoch=3,
                    valid_data_gen=valid_gen_obj,
                    validation_steps=3)
    l.plot_history()


# MAIN
if __name__ == '__main__':
    main()
