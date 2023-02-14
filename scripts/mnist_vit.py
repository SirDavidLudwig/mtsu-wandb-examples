# Set up local imports
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import wandb

import bootstrap
import common

# Tensorflow Configuration -------------------------------------------------------------------------

# Enable dynamic memory allocation on GPU for better usage monitoring
for gpu in tf.config.list_physical_devices("GPU"):
	print("Found GPU:", gpu)
	tf.config.experimental.set_memory_growth(gpu, True)

# MNIST Dataset ------------------------------------------------------------------------------------

def get_dataset():
	# Load the dataset
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
	# Normalize between 0.0 and 1.0 and expand dimensionality to (28, 28, 1)
	x_train = np.expand_dims(x_train/255.0, -1)
	x_test = np.expand_dims(x_test/255.0, -1)
	return (x_train, y_train), (x_test, y_test)

# Model Setup --------------------------------------------------------------------------------------

def get_configuration():
	parser = argparse.ArgumentParser()
	# Model Hyperparameters
	parser.add_argument("--embed_dim", type=int, default=32, help="Embedding dimension")
	parser.add_argument("--num_blocks", type=int, default=2, help="Number of transformer blocks")
	parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
	parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
	parser.add_argument("--ff_dim", type=int, default=64, help="Feed-forward layer dimensionality")
	parser.add_argument("--use_prelayernorm", type=bool, default=True, help="Use pre-layer normalization (0 or 1)")
	parser.add_argument("--num_patches", type=int, default=4, help="Number of patches to split an image into")

	# Training Hyperparameters
	parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
	parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training/evaluation")
	parser.add_argument("--optimizer", choices=list(common.OPTIMIZER_MAP.keys()), default="adam", help="Optimizer")
	parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")

	# Return the configuration
	return parser.parse_args()


def get_model(config, dataset):

	input_shape = dataset[1][0].shape[1:]
	num_labels = np.max(dataset[1][1]) + 1

	y = x = tf.keras.layers.Input(input_shape) # (28, 28, 1)

	# Image patches
	assert input_shape[0] % config.num_patches == 0, "Image size must be evenly divisible by number of patches"
	patch_size = input_shape[0] // config.num_patches
	y = tf.keras.layers.Conv2D(
		filters=config.embed_dim,
		kernel_size=patch_size,
		strides=patch_size)(y)
	y = tf.keras.layers.Reshape((-1, config.embed_dim))(y)

	# Position embeddings
	y = common.FixedPositionEmbedding(y.shape[1], config.embed_dim)(y)

	# Transformer blocks
	for _ in range(config.num_blocks):
		y = common.TransformerBlock(
			embed_dim=config.embed_dim,
			num_heads=config.num_heads,
			ff_dim=config.ff_dim,
			prenorm=config.use_prelayernorm,
			dropout_rate=config.dropout)(y)

	# Pooling and output
	y = tf.keras.layers.GlobalAveragePooling1D()(y)
	y = tf.keras.layers.Dense(config.embed_dim, activation='gelu')(y)
	y = tf.keras.layers.Dense(num_labels, activation="softmax")(y)

	model = tf.keras.Model(x, y)
	model.compile(
		loss=tf.keras.losses.SparseCategoricalCrossentropy(),
		optimizer=common.select_optimizer(config.optimizer)(config.learning_rate),
		metrics=tf.keras.metrics.SparseCategoricalAccuracy())
	model.summary()

	return model


def train(config, dataset, model):
	# Extract the dataset
	(x_train, y_train), (x_test, y_test) = dataset

	# Create the W&B callback
	wandb_callback = wandb.keras.WandbCallback()
	wandb_callback.save_model_as_artifact = False

	# Train the model
	model.fit(
		x_train,
		y_train,
		epochs=config.epochs,
		batch_size=config.batch_size,
		callbacks=[wandb_callback]
	)


def main():
	# Get the configuration
	config = get_configuration()

	# Create the W&B instance
	run = wandb.init(project="mnist_vit", config=config)

	# Train and save the model
	dataset = get_dataset()
	model = get_model(config, dataset)
	train(config, dataset, model)

	# Save the model
	model.save(os.path.join(run.dir, "model"))

	run.finish()


if __name__ == "__main__":
	main()
