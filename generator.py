from keras.datasets import mnist
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import numpy as np


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


def create_random_points(dims, num_samples):
    random_points = np.random.randn(dims * num_samples)
    random_points = random_points.reshape(num_samples, dims)
    return random_points


def create_random_img(generator, dims, num_samples):
    random_img = create_random_points(dims, num_samples)
    pred = generator.predict(random_img)
    label = np.zeros((num_samples, 1))
    return pred, label


if __name__ == "__main__":
    dims = 500
    generator = generator_model(dims)
    num_samples = 25
    fake_images, _ = create_random_img(generator, dims, num_samples)

    for i in range(num_samples):
        plt.subplot(5, 5, i + 1)
        plt.axis('off')
        plt.imshow(fake_images[i, :, :, 0], cmap='gray_r')
    plt.show()
