#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from numpy.random import randn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from keras.optimizers import Adam
import matplotlib
matplotlib.use('GTK3Cairo')
import matplotlib.pyplot as plt
 
# Creating the Generator Model with specified latent space dimension 'latent_dim'
# I.e., a 100 element vector of Gaussian random numbers
# Working without the Discriminator
def create_generator(latent_dim):
	model = Sequential()
	# Foundation for 7x7 image
	n_nodes = 128 * 7 * 7
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7, 7, 128)))
	# Upsample (Higher resolution) to 14x14
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# Upsample to 28x28
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
	return model
    
    
# Generating points in the latent space to later feed into the Generator
# for creating fake samples
def generate_latent_points(latent_dim, n_samples):
	# Generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# Reshape into batch of inputs
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input
 
# Using the generator to generate fake examples with classification y = 0.
def generate_fake_samples(g_model, latent_dim, n_samples):
	# 
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = g_model.predict(x_input)
	# Create 'fake' class labels (0)
	y = np.zeros((n_samples, 1))
	return X, y
 
# Latent space dimension
latent_dim = 100
# Create discriminator model
model = create_generator(latent_dim)
n_samples = 25
X, _ = generate_fake_samples(model, latent_dim, n_samples)
# plot the generated samples
for i in range(n_samples):
	# define subplot
	plt.plot(5, 5, 1 + i)
	# turn off axis labels
	plt.axis('off')
	# plot single image
	plt.imshow(X[i, :, :, 0], cmap='gray_r')
plt.show()
