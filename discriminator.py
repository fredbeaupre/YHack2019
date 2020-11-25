from keras import models
from keras.optimizers import Adam
from keras import layers
from keras.utils.vis_utils import plot_model


def discriminator_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2),
                            padding='same', input_shape=input_shape))
    model.add(layers.LeakyReLU(alpha=0.3))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.3))
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='tanh'))
    optimizer = Adam(lr=0.0001, beta_1=0.9)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    return model


if __name__ == "__main__":
    input_shape = (28, 28, 1)
    discriminator = discriminator_model(input_shape)
    discriminator.summary()
    plot_model(discriminator, to_file='discriminator_model.png',
               show_shapes=True, show_layer_names=True)
