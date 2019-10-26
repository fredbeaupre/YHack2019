import numpy as np
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU
 
# Define Discriminator model (here working without any generator)
def define_discriminator(in_shape=(28,28,1)):
	model = Sequential()
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model
 
# Loading MNIST images
def load_real_samples():
	(trainX, _), (_, _) = load_data()
	# Adding one channel for grayscale.
	X = np.expand_dims(trainX, axis=-1)
	# Converting to floats
	X = X.astype('float32')
	# Normalize
	X = X / 255.0
	return X
 
# Select real and random samples (images from MNIST dataset)
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = np.random.randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = np.ones((n_samples, 1))
	return X, y
 
# Generating fake images with (y = 1) for the discriminator to identify
def generate_fake_samples(n_samples):
	X = np.random.rand(28 * 28 * n_samples)
	# reshape into batch of grayscale images
	X = X.reshape((n_samples, 28, 28, 1))
	# assigning label y = 1
	y = np.zeros((n_samples, 1))
	return X, y
 
# Training the Discriminator Model
def train_discriminator(model, dataset, n_iter=100, n_batch=256):
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_iter):
		# Get real samples from MNIST
		X_real, y_real = generate_real_samples(dataset, half_batch)
		# Update discriminator on real examples
		_, real_acc = model.train_on_batch(X_real, y_real)
		# Get fake samples from MNIST
		X_fake, y_fake = generate_fake_samples(half_batch)
		# Update discriminator on fake examples
		_, fake_acc = model.train_on_batch(X_fake, y_fake)
		# Summarize performance in console
		print('>%d real=%.0f%% fake=%.0f%%' % (i+1, real_acc*100, fake_acc*100))
 
# Discriminator Model
model = define_discriminator()
# Load dataset
dataset = load_real_samples()
# Train model
train_discriminator(model, dataset)