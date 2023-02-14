import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import time
import wandb

import bootstrap
import common

print(tf.config.list_physical_devices())

# Enable dynamic memory allocation on GPU for better usage monitoring
for gpu in tf.config.list_physical_devices("GPU"):
	tf.config.experimental.set_memory_growth(gpu, True)


def get_configuration():
	parser = argparse.ArgumentParser()

	# Data Settings
	parser.add_argument("--mnist_artifact", required=True, help="The full name of the W&B MNIST dataset artifact")

	# Model Architecture
	parser.add_argument("--noise_dim", type=int, default=100, help="Ambient space embedding dimensionality")

	# Training Settings
	parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train")
	parser.add_argument("--batch_size", type=int, default=32, help="Batch size to train on")
	parser.add_argument("--g_optimizer", choices=list(common.OPTIMIZER_MAP.keys()), default="adam", help="Optimizer for the generator")
	parser.add_argument("--d_optimizer", choices=list(common.OPTIMIZER_MAP.keys()), default="adam", help="Optimizer for the discriminator")
	parser.add_argument("--g_learning_rate", type=float, default=1e-4, help="Generator learning rate")
	parser.add_argument("--d_learning_rate", type=float, default=1e-4, help="Discriminator learning rate")

	return parser.parse_args()


def load_dataset(run, config):
	# (x_train, _), _ = tf.keras.datasets.mnist.load_data()
	# x_train = x_train / 255.0
	artifact = run.use_artifact(config.mnist_artifact)
	assert artifact.metadata["scaling_method"] == "normalize", "MNIST artifact must use normalized scaling method"
	path = artifact.download()
	x_train = np.load(os.path.join(path, "mnist.npz"))["images"]
	return x_train


def create_generator(config):
	# https://www.tensorflow.org/tutorials/generative/dcgan
	y = x = tf.keras.layers.Input((config.noise_dim,))

	# Project and normalize the batch
	y = tf.keras.layers.Dense(7*7*256, use_bias=False)(y)
	y = tf.keras.layers.BatchNormalization()(y)
	y = tf.keras.layers.LeakyReLU()(y)

	# Reshape to be processed by a Conv2DTranspose
	y = tf.keras.layers.Reshape((7, 7, 256))(y)

	# Conv2DTranspose Block
	y = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same", use_bias=False)(y)
	y = tf.keras.layers.BatchNormalization()(y)
	y = tf.keras.layers.LeakyReLU()(y)

	# Conv2DTranspose Block
	y = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", use_bias=False)(y)
	y = tf.keras.layers.BatchNormalization()(y)
	y = tf.keras.layers.LeakyReLU()(y)

	y = tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding="same", use_bias=False, activation="sigmoid")(y)

	# Create the model
	generator = tf.keras.models.Model(x, y, name="Generator")
	generator.compile(
		optimizer=common.select_optimizer(config.g_optimizer)(config.g_learning_rate)
	)
	generator.summary()
	return generator


def create_discriminator(config):
	# https://www.tensorflow.org/tutorials/generative/dcgan
	# Accept any image as input
	y = x = tf.keras.layers.Input((28, 28, 1))

	# Conv2D Block
	y = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same")(y)
	y = tf.keras.layers.LeakyReLU()(y)
	y = tf.keras.layers.Dropout(0.3)(y)

	# Conv2D Block
	y = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same")(y)
	y = tf.keras.layers.LeakyReLU()(y)
	y = tf.keras.layers.Dropout(0.3)(y)

	# Project to a single output unit
	y = tf.keras.layers.Flatten()(y)
	y = tf.keras.layers.Dense(1)(y)

	# Create the model
	discriminator = tf.keras.models.Model(x, y, name="Discriminator")
	discriminator.compile(
		optimizer=common.select_optimizer(config.d_optimizer)(config.d_learning_rate)
	)
	discriminator.summary()
	return discriminator


def generator_loss(fake_output):
	# We want to fool the discriminator, so the target is 1
	return tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output, from_logits=True)

def discriminator_loss(real_output, fake_output):
	# target vs. prediction
	real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output, from_logits=True) # should predict ones
	fake_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output, from_logits=True) # should predict zeros
	return real_loss + fake_loss


@tf.function
def train_step(images, generator, discriminator):
	# Generate some noise to use for each batch
	noise = tf.random.normal([images.shape[0], generator.input_shape[1]])

	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		# Generate some random images from the generator
		gen_images = generator(noise, training=True)

		# Evaluate a batch of real and fake images against the discriminator
		real_output = discriminator(images, training=True)
		fake_output = discriminator(gen_images, training=True)

		# Compute the loss for both models
		gen_loss = generator_loss(fake_output)
		disc_loss = discriminator_loss(real_output, fake_output)

	# Compute gradients for both models
	gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
	disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

	# Apply the gradients
	generator.optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
	discriminator.optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

	return gen_loss, disc_loss, real_output, fake_output


def gen_images(generator, test_input):
	predictions = generator(test_input)
	fig = plt.figure(figsize=(6, 6))
	for i in range(predictions.shape[0]):
		plt.subplot(4, 4, i+1)
		plt.imshow(predictions[i, :, :, 0], cmap="gray")
		plt.axis("off")
	return fig


def train(run, config, dataset, generator, discriminator):

	# Generate some random noise to display the progress
	test_input = tf.random.normal([16, config.noise_dim])

	num_batches = int(np.ceil(dataset.shape[0] / config.batch_size))

	for epoch in range(config.epochs):
		start = time.time()
		total_gen_loss = 0
		total_disc_loss = 0
		gen_accuracy = 0
		disc_accuracy = 0
		for batch_index in range(num_batches):
			print(f"\rBatch: {batch_index+1}/{num_batches}", end="")
			batch = dataset[batch_index*config.batch_size:(batch_index+1)*config.batch_size]

			gen_loss, disc_loss, real_pred, fake_pred = train_step(batch, generator, discriminator)
			total_gen_loss += np.mean(gen_loss)
			total_disc_loss += np.mean(disc_loss)
			gen_accuracy += tf.where(fake_pred >= 0.5).shape[0] / config.batch_size
			disc_accuracy += (tf.where(real_pred >= 0.5).shape[0] + tf.where(fake_pred < 0.5).shape[0]) / (2*config.batch_size)

		total_gen_loss /= num_batches
		total_disc_loss /= num_batches
		gen_accuracy /= num_batches
		disc_accuracy /= num_batches

		fig = gen_images(generator, test_input)
		tf.print(f"\rEpoch {epoch} completed. Total epoch time: {time.time() - start:0.2} seconds.")

		run.log({
			"epoch": epoch + 1,
			"generator_accuracy": gen_accuracy,
			"discriminator_accuracy": disc_accuracy,
			"generator_loss": float(total_gen_loss),
			"discriminator_loss": float(total_disc_loss),
			"evaluation": fig
		})


def main():

	# Get the configuration
	config = get_configuration()

	# Create the W&B run
	run = wandb.init(project="mnist_dcgan", config=config)

	# Load the dataset
	dataset = load_dataset(run, config)

	# Create and train the generator + discriminator
	generator = create_generator(config)
	discriminator = create_discriminator(config)
	train(run, config, dataset, generator, discriminator)

	# Save the generator
	save_path = os.path.join(run.dir, "generator")
	generator.save(save_path)

	# Log the generator model as an artifact
	artifact = wandb.Artifact("mnist-generator", type="model")
	artifact.add_dir(save_path)
	run.log_artifact(artifact)

	run.finish()

if __name__ == "__main__":
	main()
