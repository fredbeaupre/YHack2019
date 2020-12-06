from keras import layers
from keras import models
from keras.datasets import mnist
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils.vis_utils import plot_model


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


def create_fake_images(generator, dims, num_samples):
    random_img = create_random_points(dims, num_samples)
    pred = generator.predict(random_img)
    label = np.zeros((num_samples, 1))
    return pred, label


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


def gan_model(generator, discriminator):
    discriminator.trainable = False
    model = models.Sequential()
    model.add(generator)
    model.add(discriminator)
    optimizer = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model


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


def save_image(image, epoch, n=10):
    for i in range(n*n):
        plt.subplot(n, n, i+1)
        plt.axis('off')
        plt.imshow(image[i, :, :, 0], cmap='gray_r')
    file_name = 'plot_epoch%03d.png' % (epoch+1)
    plt.savefig(file_name)
    plt.close()


def recap(epoch, generator, discriminator, dataset, dims, num_samples=100):
    real_x, real_y = select_real_digits(dataset, num_samples)
    _, real_accuracy = discriminator.evaluate(real_x, real_y, verbose=0)
    fake_x, fake_y = create_fake_images(generator, dims, num_samples)
    _, fake_accuracy = discriminator.evaluate(fake_x, fake_y, verbose=0)
    print('Accuracy on real samples: %.0f%%; Accuracy onf fake samples: %.0f%%' % (
        real_accuracy*100, fake_accuracy*100))
    save_image(fake_x, epoch)
    file_name = 'generator_%03d.h5' % (epoch+1)
    generator.save(file_name)


def train_gan(generator, discriminator, gan, dataset, dims, num_epochs=100, batch_size=256):
    batches_per_epoch = int(dataset.shape[0] / batch_size)
    half_batch_size = int(batch_size / 2)

    for i in range(num_epochs):
        for j in range(batches_per_epoch):
            real_x, real_y = select_real_digits(dataset, half_batch_size)
            fake_x, fake_y = create_fake_images(
                generator, dims, half_batch_size)
            X, y = np.vstack((real_x, fake_x)), np.vstack((real_y, fake_y))
            discriminator_loss, _ = discriminator.train_on_batch(X, y)
            gan_x = create_random_points(dims, batch_size)
            gan_y = np.ones((batch_size, 1))
            generator_loss = gan.train_on_batch(gan_x, gan_y)
            print('>>> Epoch %d, Batch %d/%d: Generator Loss=%.3f, Discriminator Loss=%.3f' %
                  (i+1, j+1, batches_per_epoch, generator_loss, discriminator_loss))
        if (i+1) % 5 == 0:
            recap(i, generator, discriminator, dataset, dims)


if __name__ == "__main__":
    dims = 500
    discriminator = discriminator_model()
    generator = generator_model(dims)
    gan = gan_model(generator, discriminator)
    mnist_data = preprocess_mnist()
    train_gan(generator, discriminator, gan, mnist_data, dims)
