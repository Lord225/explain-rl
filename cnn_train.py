import tensorflow as tf        
from tensorflow import keras   
import keras.layers            
import keras.activations       
import keras.optimizers        
import keras                   
import matplotlib.pyplot as plt
import VisualTransformer       


def run_experiment():
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # load the dataset using keras

    train, test = keras.datasets.cifar10.load_data()
    x_train, y_train = train

    y_train = y_train.astype("float32")

    # # pad images to 32x32
    # x_train = np.pad(
    #     x_train, ((0, 0), (2, 2), (2, 2)), mode="constant", constant_values=0
    # )

    x_train = x_train / 255.0

    x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)

    # convert to tf dataset for better performance and augmentation (include y_train)
    x_train = tf.data.Dataset.from_tensor_slices(x_train)

    # add augmentation
    random_augmentation = keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomTranslation(0.2, 0.2),
        ]
    )

    x_train = x_train.repeat(5).map(lambda x: random_augmentation(x, training=True), AUTOTUNE)

    # add y
    y_train = tf.data.Dataset.from_tensor_slices(y_train).repeat(5)

    # zip x and y
    train = tf.data.Dataset.zip((x_train, y_train))
    # batch, repeat, prefetch
    train = train.batch(128).prefetch(AUTOTUNE)

    x_test, y_test = test

    x_test = x_test / 255.0

    x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)

    y_test = y_test.astype("float32")

    # use CNN
    model = tf.keras.Sequential(
        # Simple ~200k param cnn net
        [
            keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3), padding="same"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(8, (2, 2), activation="relu", padding="same"),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )

    optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    # summarize the model
    model.build((None, 32, 32, 3))
    model.summary()

    _ = model.predict(x_test[:1])

    callbacks = [
        keras.callbacks.LearningRateScheduler(lambda epoch: 0.001 * 0.92 ** (epoch))
    ]

    model.fit(train, batch_size=128, epochs=10, callbacks=callbacks)
    _, accuracy = model.evaluate(x_test, y_test)

    plt.imshow(x_test[:1].reshape(32, 32, 3), cmap="gray")

    # save
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")


    model.save("cnn_cifar10")


if __name__ == "__main__":
    run_experiment()
