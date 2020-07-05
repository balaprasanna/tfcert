from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers, Model
import tensorflow as tf

input_shape = (150,150,3)
filepath = "/home/prasanna/Downloads/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
pre_trained_model = InceptionV3(input_shape=input_shape, include_top=False, weights=None)
pre_trained_model.load_weights(filepath)

for layer in pre_trained_model.layers:
    print("Layer name: {}".format(layer.name))
    layer.trainable = False

last_layer = pre_trained_model.get_layer("mixed8")
last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation="relu")(x)
final_output = layers.Dense(1, activation="sigmoid")(x)

model = Model(pre_trained_model.input, final_output)
model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['acc'])

pre_trained_model.summary()
