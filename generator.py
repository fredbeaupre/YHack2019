from keras.datasets import mnist
from keras import models
from keras import layers
import matplotlib.pyplot as plt


def generator_model(dims):
    model = models.Sequential()
    n_nodes = 128 * 7 * 7
    model.add(layers.Dense(n_nodes, input_dim=dims))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((7, 7, 128)))
    # upsampling
    model.add(layers.Conv2DTranspose(
        128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # upsampling
    model.add(layers.Conv2DTranspose(
        128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(1, (7, 7), activation='sigmoid', padding='same'))
    return model


if __name__ == "__main__":
    dims = 500
    generator = generator_model(dims)
    generator.summary()
