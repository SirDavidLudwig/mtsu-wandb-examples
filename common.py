# Much of this code is reference from https://github.com/DLii-Research/deep-learning-dna

import tensorflow as tf
import tensorflow.keras as keras

# A simple lookup for optimizers
OPTIMIZER_MAP = {
    "adam": tf.keras.optimizers.Adam,
    "nadam": tf.keras.optimizers.Nadam,
    "rmsprop": tf.keras.optimizers.RMSprop,
    "sgd": tf.keras.optimizers.SGD
}

def select_optimizer(optimizer: str) -> tf.keras.optimizers.Optimizer:
    """
    Fetch an optimizer object by string name.
    Returns: Optimizer
    """
    if optimizer not in OPTIMIZER_MAP:
            raise Exception("Unknown optimizer:", optimizer)
    return OPTIMIZER_MAP[optimizer]

# Layers -------------------------------------------------------------------------------------------

class BaseTransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, ff_activation="gelu", gating=None, dropout_rate=0.1, prenorm=False, **kwargs):
        super().__init__(**kwargs)
        self.ffn = keras.Sequential(
            [keras.layers.Dense(ff_dim, activation=ff_activation),
             keras.layers.Dense(embed_dim),]
        )
        # Input parameters
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.ff_activation = ff_activation
        self.gating = gating
        self.dropout_rate = dropout_rate
        self.prenorm = prenorm

        # Internal layers
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)
        self.att = self.create_attention_layer(embed_dim, num_heads)

        self.supports_masking = True

    def create_attention_layer(self, embed_dim, num_heads):
        raise NotImplemented()

    def att_prenorm(self, inputs, training):
        inputs_norm = self.layernorm1(inputs)
        attn_output = self.att(inputs_norm, inputs_norm)
        attn_output = self.dropout1(attn_output, training=training)
        attn_output = inputs + attn_output

        ffn_norm = self.layernorm2(attn_output)
        ffn_output = self.ffn(ffn_norm)
        ffn_output = self.dropout2(ffn_output, training=training)

        return attn_output + ffn_output

    def att_postnorm(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def call(self, inputs, training):
        if self.prenorm:
            return self.att_prenorm(inputs, training)
        return self.att_postnorm(inputs, training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "ff_activation": self.ff_activation,
            "gating": self.gating,
            "dropout_rate": self.dropout_rate,
            "prenorm": self.prenorm
        })
        return config


class TransformerBlock(BaseTransformerBlock):
    def __init__(self, *args, use_vaswani_mha=False, **kwargs):
        self.use_vaswani_mha = use_vaswani_mha
        super().__init__(*args, **kwargs)

    def create_attention_layer(self, embed_dim, num_heads):
        if self.use_vaswani_mha:
            return VaswaniMultiHeadAttention(num_heads=num_heads, embed_dim=embed_dim)
        return keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)

    def build(self, input_shape):
        if not self.use_vaswani_mha:
            self.att._build_from_signature(input_shape, input_shape)
        return super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "use_vaswani_mha": self.use_vaswani_mha
        })
        return config


class FixedPositionEmbedding(keras.layers.Layer):
    def __init__(self, length, embed_dim):
        super().__init__()
        self.length = length
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.positions = self.add_weight(
            shape=(self.length, self.embed_dim),
            initializer="uniform",
            trainable=True,
            name="position_embeddings")

    def call(self, x):
        return x + self.positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "length": self.length,
            "embed_dim": self.embed_dim
        })
        return config