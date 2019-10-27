import numpy as np
from numpy import vstack
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib
matplotlib.use('GTK3Cairo')
import matplotlib.pyplot as plt
 
# Create the generator model
def create_generator(latent_dim):
	model = models.Sequential()
	n_nodes = 128 * 7 * 7
	model.add(layers.Dense(n_nodes, input_dim=latent_dim))
	model.add(layers.LeakyReLU(alpha=0.2))
	model.add(layers.Reshape((7, 7, 128)))
	# upsample (To higher resolution) to 14x14
	model.add(layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(layers.LeakyReLU(alpha=0.2))
	model.add(layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(layers.LeakyReLU(alpha=0.2))
	model.add(layers.Conv2D(1, (7,7), activation='sigmoid', padding='same'))
	return model

# Create the discriminator model
def create_discriminator(in_shape=(28,28,1)):
	model = models.Sequential()
	model.add(layers.Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
	model.add(layers.LeakyReLU(alpha=0.2))
	model.add(layers.Dropout(0.4))
	model.add(layers.Conv2D(64, (3,3), strides=(2, 2), padding='same'))
	model.add(layers.LeakyReLU(alpha=0.2))
	model.add(layers.Dropout(0.4))
	model.add(layers.Flatten())
	model.add(layers.Dense(1, activation='sigmoid'))
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

 
# Define the model of the combined generator and discriminator.
def GANmodel(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect them
	model = models.Sequential()
	# add generator
	model.add(g_model)
	# add the discriminator
	model.add(d_model)
	# compile model
	optimizer = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=optimizer)
	return model
 
# Load images from MNIST data set ('real' images) for training
def load_MNIST():
	# load mnist dataset
	(trainX, _), (_, _) = load_data()
	# expand to 3d, e.g. add channels dimension
	X = np.expand_dims(trainX, axis=-1)
	# convert from unsigned ints to floats
	X = X.astype('float32')
	# scale from [0,255] to [0,1]
	X /= 255.0
	return X
 
# Choose random real samples
def generate_MNIST(dataset, n_samples):
	# choose random instances
	inst = np.random.randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[inst]
	y = np.ones((n_samples, 1))
	return X, y
 
# Generate points from the latent space to feed into the generator
def generate_latent_points(latent_dim, n_samples):
	x_input = np.random.randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input
 
# Using generator to create fake images, with y = 0.
def generate_fake(g_model, latent_dim, n_samples):
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = g_model.predict(x_input)
	y = np.zeros((n_samples, 1))
	return X, y
 
# Creating plot of generated images after some number n of epeochs
def save_plot(examples, epoch, n=10):
	for i in range(n * n):
		plt.subplot(n, n, 1 + i)
		plt.axis('off')
		plt.imshow(examples[i, :, :, 0], cmap='gray_r')
	filename = 'generated_plot_e%03d.png' % (epoch+1)
	plt.savefig(filename)
	plt.close()
 
# Evaluating performance
def performance(epoch, gen_model, disc_model, dataset, latent_dim, n_samples=100):
	# Real samples
	X_real, y_real = generate_MNIST(dataset, n_samples)
	# Evaluate discriminator on real examples
	_, acc_real = disc_model.evaluate(X_real, y_real, verbose=0)
	# Fake examples
	X_fake, y_fake = generate_fake(gen_model, latent_dim, n_samples)
	# Evaluate discriminator on fake examples
	_, acc_fake = disc_model.evaluate(X_fake, y_fake, verbose=0)
	# Output discriminator performance summary to console
	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
	save_plot(X_fake, epoch)
	filename = 'generator_model_%03d.h5' % (epoch + 1)
	gen_model.save(filename)
 
# Training generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=256):
	bat_per_epo = int(dataset.shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	# Iterate over epochs
	for i in range(n_epochs):
		# Iterate over training set
		for j in range(bat_per_epo):
			# Real samples
			X_real, y_real = generate_MNIST(dataset, half_batch)
			# Fake samples
			X_fake, y_fake = generate_fake(g_model, latent_dim, half_batch)
			# Training set for discriminator
			X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
			# Update discriminator weights
			d_loss, _ = d_model.train_on_batch(X, y)
			# Points from latent space
			X_gan = generate_latent_points(latent_dim, n_batch)
			# Inverted labels
			y_gan = np.ones((n_batch, 1))
			# Update generator from discrimination error
			g_loss = gan_model.train_on_batch(X_gan, y_gan)
			# Print summary to console
			print('Epoch %d;  %d/%d, d=%.3f;  g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))
		if (i+1) % 5 == 0:
			performance(i, g_model, d_model, dataset, latent_dim)
 
# size of the latent space
latent_dim = 100
# create the discriminator
d_model = create_discriminator()
# create the generator
g_model = create_generator(latent_dim)
# create the gan
gan_model = GANmodel(g_model, d_model)
# load image data
dataset = load_MNIST()
# train model
train(g_model, d_model, gan_model, dataset, latent_dim)