import tensorflow as tf
from tensorflow import keras
import numpy as np
import keras.layers
import keras.activations
import keras.optimizers
import keras
import VisualTransformer 
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

    # plt.show()


    model = VisualTransformer.ViT(
        image_size=(32, 32, 1),
        patch_size=4,
        num_layers=2,
        hidden_size=16,
        num_heads=8,
        name="vit",
        mlp_dim=16,
        classes=10,
        dropout=0.15,
        activation="linear",
        representation_size=16,
    )


    optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    # summarize the model
    model.build((None, 32, 32, 1))
    model.summary()

    _ = model.predict(x_train[:1])

    # Call the model on some input data to define the input shape
    attention = VisualTransformer.attention_map(model, x_train[0])
    plt.matshow(attention.reshape(32, 32), cmap="viridis")
    plt.show()


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

    model.fit(x_train, y_train, batch_size=256, epochs=150, callbacks=callbacks)
    _, accuracy = model.evaluate(x_test, y_test)

    # plot x_train[:1]
    plt.imshow(x_train[:1].reshape(32, 32, 1), cmap="gray")
    
    # attention map is of shape (1, 1, 2, 64, 64)
    attention = VisualTransformer.attention_map(model, x_train[0])
    # plot attention maps ((32, 32, 1))
    plt.matshow(attention.reshape(32, 32), cmap="viridis")
    plt.show()

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
