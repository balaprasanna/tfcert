# from modelutils.Base import BaseModel
# from modelutils.Callbacks import StopTrainingCb
from modelutils import BaseModel, StopTrainingCb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class PreprocessTFLayer(tf.keras.layers.Layer):
    def __init__(self, name="preprocess_tf", **kwargs):
        super(PreprocessTFLayer, self).__init__(name=name, **kwargs)
        self.preprocess = tf.keras.applications.vgg19.preprocess_input

    def call(self, input):
        return self.preprocess(input)

    def get_config(self):
        config = super(PreprocessTFLayer, self).get_config()
        return config


class CatsVsDogs(BaseModel):
    def __init__(self, input_shape, n_classes):
        super().__init__(input_shape, n_classes)
        self.input_shape = input_shape
        self.optimizer = keras.optimizers.Adam()
        self.loss = keras.losses.categorical_crossentropy
        self.n_classes = n_classes
        self.metrics = ["acc"]
        self.model = self.model_arch()

    def model_arch(self):
        self.base_model = tf.keras.applications.vgg19.VGG19(
            input_shape=self.input_shape,
            include_top=False,
            weights=None
        )
        filepath = "/home/prasanna/Downloads/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"
        self.base_model.load_weights(filepath)

        for layer in self.base_model.layers:
            print("Layer name: {}".format(layer.name))
            layer.trainable = False

        last_layer = self.base_model.get_layer("block5_conv4")
        last_output = last_layer.output

        # x = keras.layers.GlobalAveragePooling2D()(last_output),
        x = tf.keras.layers.Flatten()(last_output),
        x = tf.keras.layers.Dense(1024, activation="relu")(x)
        final_output = tf.keras.layers.Dense(6, activation="softmax")(x)

        model = tf.keras.Model(self.base_model.input, final_output)
        return model
        # return tf.keras.Sequential(
        #     layers=[
        #         tf.keras.Input(shape=self.input_shape, name="input"),
        #         # PreprocessTFLayer(),
        #         self.base_model,
        #         keras.layers.GlobalAveragePooling2D(),
        #         keras.layers.Dense(units=1024, activation=tf.nn.relu),
        #         keras.layers.Dense(units=64, activation=tf.nn.relu),
        #         keras.layers.Dense(units=self.n_classes, activation=tf.nn.softmax)
        #     ]
        # )



def main(*args):
    input_shape = (150, 150, 3)
    n_classes = 6
    train_dir = "/opt/localtmp/dataset/flowers-recognition/flowers"
    train_imgae_gen = ImageDataGenerator(rescale=1/255.)
    train_imgae_gen_obj = train_imgae_gen.flow_from_directory(
        train_dir,
        target_size=input_shape[:-1],
        class_mode="categorical",
        batch_size=32)

    cb = StopTrainingCb()
    model = CatsVsDogs(input_shape, n_classes)
    model.register_callbacks(cb)
    model.compile()
    model.fit_generator(train_imgae_gen_obj)


if __name__ == '__main__':
    main()
