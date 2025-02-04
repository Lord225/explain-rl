from gc import callbacks
from math import pi
import tensorflow as tf
from tensorflow import keras
import numpy as np
import keras.layers
import keras.activations
import keras.optimizers

import keras
from keras import layers

class MLP(layers.Layer):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.dense_layers = [
            layers.Dense(units, activation=keras.activations.gelu)
            for units in hidden_units
        ]
        self.dropout_layers = [layers.Dropout(dropout_rate) for _ in hidden_units]

    def call(self, inputs):
        x = inputs
        for dense, dropout in zip(self.dense_layers, self.dropout_layers):
            x = dense(x)
            x = dropout(x)
        return x

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_units": self.hidden_units, "dropout_rate": self.dropout_rate})
        return config

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):

        input_shape = tf.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(
            patches, (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.expand_dims(
            tf.range(start=0, limit=self.num_patches, delta=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    
class VisionTransformer(keras.Model):
    def __init__(self, input_shape, num_classes, patch_size, num_patches, projection_dim, 
                 transformer_layers, num_heads, transformer_units,):
        super().__init__()
        self.input_shape_config = input_shape  # Store input shape for config
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.projection_dim = projection_dim  # Store projection_dim for config
        self.transformer_layers = transformer_layers
        self.num_heads = num_heads
        self.transformer_units = transformer_units

        self.patches = Patches(patch_size)
        self.encoder = PatchEncoder(num_patches, projection_dim)

        self.transformer_blocks = []
        for _ in range(transformer_layers):
            self.transformer_blocks.append([
                layers.LayerNormalization(epsilon=1e-6),
                layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.25 ),
                layers.Add(),
                layers.LayerNormalization(epsilon=1e-6),
                MLP(hidden_units=transformer_units, dropout_rate=0.1),
                layers.Add()
            ])
        
        self.representation_layer = layers.LayerNormalization(epsilon=1e-6)
        self.flatten = layers.Flatten()
        self.classifier = layers.Dense(num_classes)

    def call(self, inputs, training=False, return_attention_scores=False):
        x = self.patches(inputs)
        #print(x.shape)
        x = self.encoder(x)
        #print(x.shape)
        attention_scores = []
        for norm1, attn, add1, norm2, mlp, add2 in self.transformer_blocks:
            attn_input = norm1(x)
            if return_attention_scores:
                attn_output, weights  = attn(attn_input, attn_input, return_attention_scores=return_attention_scores)
                attention_scores.append(weights)
            else:
                attn_output = attn(attn_input, attn_input)
            x = add1([attn_output, x])
            
            mlp_input = norm2(x)
            mlp_output = mlp(mlp_input)
            x = add2([mlp_output, x])

        # print(x.shape) # (None, 64, 32),
        # It means: (batch_size, num_patches, projection_dim)
        
        x = x[:, 0, :]
        x = self.representation_layer(x)
        x = self.flatten(x)
        if return_attention_scores:
            return self.classifier(x), tf.stack(attention_scores)
        return self.classifier(x)
    
    def compute_attention_map(self, image):
        _, attention_map = self.call(image, return_attention_scores=True)
        return attention_map
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "input_shape": self.input_shape_config,
            "num_classes": self.num_classes,
            "patch_size": self.patch_size,
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
            "transformer_layers": self.transformer_layers,
            "num_heads": self.num_heads,
            "transformer_units": self.transformer_units,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

import matplotlib.pyplot as plt

def run_experiment():
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # load the dataset using keras

    train, test = keras.datasets.mnist.load_data()
    x_train, y_train = train

    y_train = y_train.astype("float32")

    # pad images to 32x32
    x_train = np.pad(
        x_train, ((0, 0), (2, 2), (2, 2)), mode="constant", constant_values=0
    )
    print(x_train.shape)

    x_train = x_train / 255.0

    x_train = x_train.reshape(x_train.shape[0], 32, 32, 1)

    x_test, y_test = test
    
    x_test = x_test.astype("float32")

    x_test = np.pad(
        x_test, ((0, 0), (2, 2), (2, 2)), mode="constant", constant_values=0
    )

    x_test = x_test / 255.0

    x_test = x_test.reshape(x_test.shape[0], 32, 32, 1)
    y_test = y_test.astype("float32")
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 1)
    plt.figure(figsize=(4, 4))
    image = x_train[np.random.choice(range(x_train.shape[0]))]
    plt.imshow(image.reshape(32, 32, 1), cmap="gray")
    plt.axis("off")
    image_size = 32
    patch_size = 4
    patches = Patches(patch_size)(image.reshape(1, image_size, image_size, 1))
    print(f"Image size: {image_size} X {image_size}")
    print(f"Patch size: {patch_size} X {patch_size}")
    print(f"Patches per image: {patches.shape[1]}")
    print(f"Elements per patch: {patches.shape[-1]}")

    # n = int(np.sqrt(patches.shape[1]))
    # plt.figure(figsize=(4, 4))
    # for i, patch in enumerate(patches[0]):
    #     ax = plt.subplot(n, n, i + 1)
    #     patch_img = tf.reshape(patch, (patch_size, patch_size))
    #     plt.imshow(patch_img.numpy().astype("uint8"))
    #     plt.axis("off")

    # plt.show()

    # model = create_vit_classifier(
    #     input_shape=(32, 32, 1),
    #     num_classes=10,
    #     patch_size=4,
    #     num_patches=64,
    #     projection_dim=16,
    #     transformer_layers=1,
    #     num_heads=2,
    #     transformer_units=[16],
    #     mlp_head_units=[32],
    # )

    model = VisionTransformer(
        input_shape=(image_size, image_size, 1),        
        num_classes=10,               
        patch_size=patch_size,      
        num_patches=64,           
        projection_dim=16,          
        transformer_layers=2,       
        num_heads=16,           
        transformer_units=[32, 16],     
    )

    attention_mask = model.compute_attention_map(x_train[:1])
    print(attention_mask)
    print(attention_mask.shape)

    optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    # summarize the model
    model.build((None, 32, 32, 1))
    model.summary()

    # for each image resize it to 32x32
    # x_train = tf.image.resize(x_train[..., tf.newaxis], (32, 32)).numpy()
    # x_test = tf.image.resize(x_test[..., tf.newaxis], (32, 32)).numpy()
    # print(x_train.shape)

    # try getting attention weights
    #attentions = model.get_attentions(x_train[:1])

    #print(attentions)

    callbacks = [
        keras.callbacks.LearningRateScheduler(lambda epoch: 0.001 * 0.92 ** (epoch))
    ]

    model.fit(x_train, y_train, batch_size=128, epochs=150, callbacks=callbacks)
    _, accuracy = model.evaluate(x_test, y_test)

    # plot x_train[:1]
    plt.imshow(x_train[:1].reshape(32, 32, 1), cmap="gray")
    
    # attention map is of shape (1, 1, 2, 64, 64)
    attentions = model.compute_attention_map(x_train[:1])
    # plot attention maps
    plt.matshow(attentions[0][0][0], cmap="viridis")

    # attentions = attentions[0][0]

    # # print(attentions.shape) (2, 64, 64)
    # # dimensions are (num_heads, num_patches, num_patches)
    # # calucalte average attention across heads and reshape to 8x8 grid
    # avg_attentions = tf.reduce_mean(attentions, axis=0)
    # # shape is now (64, 64)
    # # average attention for each patch
    # avg_attentions = tf.reduce_mean(avg_attentions, axis=0)
    # # shape is now (64,)
    # # reshape to 8x8 grid
    # avg_attentions = tf.reshape(avg_attentions, (8, 8))
    # # plot
    # plt.matshow(avg_attentions, cmap="viridis")

    # plt.show()

    # print(attentions)

    # save
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")


    model.save("mints10")


if __name__ == "__main__":
    run_experiment()
