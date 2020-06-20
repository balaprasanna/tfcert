from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = "/opt/localtmp/train"
valid_dir = "/opt/localtmp/valid"
image_gen = ImageDataGenerator(rescale=1./255)
image_gen.flow_from_directory(train_dir,
                              target_size=(300,300),
                              batch_size=128,
                              class_mode="binary"
                              )

valid_image_gen = ImageDataGenerator(rescale=1./255)
valid_image_gen.flow_from_directory(valid_dir,
                              target_size=(300,300),
                              batch_size=38,
                              class_mode="binary"
                              )

if __name__ == '__main__':
    print(image_gen)