from keras import models
from keras.optimizers import Adam
from keras import layers
from keras.utils.vis_utils import plot_model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np


def discriminator_model(input_shape=(28, 28, 1)):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2),
                            padding='same', input_shape=input_shape))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.4))
    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.4))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    optimizer = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    return model


def preprocess_mnist():
    (x_train, _), (_, _) = mnist.load_data()
    x_train_graychannel = np.expand_dims(x_train, axis=-1)
    x_train_graychannel = x_train_graychannel.astype('float32')
    x_train_graychannel = x_train_graychannel/255.0
    return x_train_graychannel


def select_real_digits(dataset, num_digits):
    index = np.random.randint(0, dataset.shape[0], num_digits)
    digits = dataset[index]
    label = np.ones((num_digits, 1))
    return digits, label


def create_fake_digits(num_digits):
    fake_digits = np.random.rand(28 * 28 * num_digits)
    fake_digits = fake_digits.reshape((num_digits, 28, 28, 1))
    label = np.zeros((num_digits, 1))
    return fake_digits, label


def train(model, dataset, num_iters=100, batch_size=256):
    half_batch_size = int(batch_size / 2)
    for i in range(num_iters):
        # training on instances y = 1
        real_x, real_y = select_real_digits(dataset, half_batch_size)
        _, real_accuracy = model.train_on_batch(real_x, real_y)

        # training on instances y = 0
        fake_x, fake_y = create_fake_digits(half_batch_size)
        _, fake_accuracy = model.train_on_batch(fake_x, fake_y)
        # output performance
        print(f'Iteration {i + 1}')
        print(f'Accuracy on real samples: {real_accuracy*100}')
        print(f'Accuracy on fake samples: {fake_accuracy*100}\n\n')


if __name__ == "__main__":
    discriminator = discriminator_model()
    mnist_data = preprocess_mnist()
    train(discriminator, mnist_data)
