from typing import Optional
import tensorflow as tf
import numpy as np
import keras


@tf.keras.utils.register_keras_serializable()
class ClassToken(tf.keras.layers.Layer):
    def build(self, input_shape):
        cls_init = tf.zeros_initializer()
        self.hidden_size = input_shape[-1]
        self.cls = tf.Variable(
            name="cls",
            initial_value=cls_init(shape=(1, 1, self.hidden_size), dtype=tf.float32),
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_broadcasted = tf.cast(
            tf.broadcast_to(self.cls, [batch_size, 1, self.hidden_size]),
            dtype=inputs.dtype,
        )
        return tf.concat([cls_broadcasted, inputs], 1)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class AddPositionEmbs(tf.keras.layers.Layer):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def build(self, input_shape):
        assert (
            len(input_shape) == 3
        ), f"Number of dimensions should be 3, got {len(input_shape)}"
        self.pe = tf.Variable(
            name="pos_embedding",
            initial_value=tf.random_normal_initializer(stddev=0.06)(
                shape=(1, input_shape[1], input_shape[2])
            ),
            dtype="float32",
            trainable=True,
        )

    def call(self, inputs):
        return inputs + tf.cast(self.pe, dtype=inputs.dtype)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, *args, num_heads, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        num_heads = self.num_heads
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {hidden_size} should be divisible by number of heads = {num_heads}"
            )
        self.hidden_size = hidden_size
        self.projection_dim = hidden_size // num_heads
        self.query_dense = tf.keras.layers.Dense(hidden_size, name="query")
        self.key_dense = tf.keras.layers.Dense(hidden_size, name="key")
        self.value_dense = tf.keras.layers.Dense(hidden_size, name="value")
        self.combine_heads = tf.keras.layers.Dense(hidden_size, name="out")

    # pylint: disable=no-self-use
    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], score.dtype)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.hidden_size))
        output = self.combine_heads(concat_attention)
        return output, weights

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    """Implements a Transformer block."""

    def __init__(self, *args, num_heads, mlp_dim, dropout, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

    def build(self, input_shape):
        self.att = MultiHeadSelfAttention(
            num_heads=self.num_heads,
            name="MultiHeadDotProductAttention_1",
        )
        self.mlpblock = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self.mlp_dim,
                    activation="gelu",
                    name=f"{self.name}/Dense_0",
                ),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.Dense(input_shape[-1], name=f"{self.name}/Dense_1"),
                tf.keras.layers.Dropout(self.dropout),
            ],
            name="MlpBlock_3",
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_0"
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_2"
        )
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout)

    def call(self, inputs, training):
        x = self.layernorm1(inputs)
        x, weights = self.att(x)
        x = self.dropout_layer(x, training=training)
        x = x + inputs
        y = self.layernorm2(x)
        y = self.mlpblock(y)
        return x + y, weights

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "dropout": self.dropout,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

class ViT(tf.keras.Model):
    def __init__(
        self,
        image_size: tuple[int, int, int],
        patch_size: int,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        name: str,
        mlp_dim: int,
        classes: int,
        dropout=0.1,
        activation="linear",
        representation_size: Optional[int] = None,
        preprocess=None,
    ):
        super(ViT, self).__init__(name=name)

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.classes = classes
        self.dropout = dropout
        self.activation = activation
        self.representation_size = representation_size

        if preprocess is not None:
            self.preprocess = preprocess
        else:
            self.preprocess = None

        self.embedding = tf.keras.layers.Conv2D(
            filters=self.hidden_size,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid",
            name="embedding",
            input_shape=image_size,
        )
        
        self.class_token = ClassToken(name="class_token")
        self.pos_embed = AddPositionEmbs(name="Transformer/posembed_input")
        self.encoder_layers = [
            TransformerBlock(
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout=self.dropout,
                name=f"Transformer/encoderblock_{n}",
            ) for n in range(self.num_layers)
        ]
        
        self.encoder_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="Transformer/encoder_norm"
        )
        
        self.extract_token = tf.keras.layers.Lambda(lambda v: v[:, 0], name="ExtractToken")
        
        if self.representation_size is not None:
            self.pre_logits = tf.keras.layers.Dense(self.representation_size, name="pre_logits", activation="tanh")
        else:
            self.pre_logits = None
        
        self.head = tf.keras.layers.Dense(self.classes, name="head", activation=self.activation)

        assert (image_size[0] % patch_size == 0) and (image_size[1] % patch_size == 0), "image_size must be a multiple of patch_size"
        

    def call(self, inputs):
        if self.preprocess is not None:
            x = self.preprocess(inputs)
        else:
            x = inputs
        
        x = self.embedding(x)
        x = tf.keras.layers.Reshape((x.shape[1] * x.shape[2], x.shape[3]))(x)
        x = self.class_token(x)
        x = self.pos_embed(x)
        
        for layer in self.encoder_layers:
            x, _ = layer(x)
        
        x = self.encoder_norm(x)
        x = self.extract_token(x)
        
        if self.pre_logits is not None:
            x = self.pre_logits(x)
        
        x = self.head(x)
        return x
    
    def get_attentions(self, inputs):
        x = self.embedding(inputs)
        x = tf.keras.layers.Reshape((x.shape[1] * x.shape[2], x.shape[3]))(x)
        x = self.class_token(x)
        x = self.pos_embed(x)
        
        attentions = []
        for layer in self.encoder_layers:
            _, weights = layer(x)
            attentions.append(weights)

        return np.stack(attentions).squeeze(axis=1)


    def get_config(self):
        config = super().get_config()
        config.update({
            "name": self.name,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "classes": self.classes,
            "dropout": self.dropout,
            "activation": self.activation,
            "representation_size": self.representation_size
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    
import cv2

def attention_map(model: ViT, image: np.ndarray):
    img_height, img_width = model.image_size[:-1]
    channel = model.image_size[-1]
    grid_size = img_height // model.patch_size

    # Prepare the input
    X = cv2.resize(image, (img_height, img_width)).reshape(1, img_height, img_width, channel)

    weights = model.get_attentions(X)
    print(weights.shape)
    num_layers = weights.shape[0]
    num_heads = weights.shape[1]
    reshaped = weights.reshape(
        (num_layers, num_heads, grid_size**2 + 1, grid_size**2 + 1)
    )

    # From Appendix D.6 in the paper ...
    # Average the attention weights across all heads.
    reshaped = reshaped.mean(axis=1)

    # From Section 3 in https://arxiv.org/pdf/2005.00928.pdf ...
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    reshaped = reshaped + np.eye(reshaped.shape[1])
    reshaped = reshaped / reshaped.sum(axis=(1, 2))[:, np.newaxis, np.newaxis]

    # Recursively multiply the weight matrices
    v = reshaped[-1]
    for n in range(1, len(reshaped)):
        v = np.matmul(v, reshaped[-1 - n])

    # Attention from the output token to the input space.
    mask = v[0, 1:].reshape(grid_size, grid_size)
    mask = cv2.resize(mask / mask.max(), (image.shape[1], image.shape[0]))[
        ..., np.newaxis
    ]
    return mask