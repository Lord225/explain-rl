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

    x_test, y_test = test

    x_test = x_test / 255.0

    x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)

    y_test = y_test.astype("float32")

    # pad images to 32x32
    # x_test = np.pad(
    #     x_test, ((0, 0), (2, 2), (2, 2)), mode="constant", constant_values=0
    # )

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

    model = VisualTransformer.ViT(
        image_size=(32, 32, 3),
        patch_size=4,
        num_layers=6,
        hidden_size=64,
        num_heads=4,
        name="vit",
        mlp_dim=64,
        classes=10,
        dropout=0.1,
        activation="linear",
        representation_size=32,
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

    _ = model.predict(x_train[:1])

    # Call the model on some input data to define the input shape
    attention = VisualTransformer.attention_map(model, x_train[0])
    plt.matshow(attention.reshape(32, 32), cmap="viridis")
    plt.show()

    callbacks = [
        keras.callbacks.LearningRateScheduler(lambda epoch: 0.001 * 0.92 ** (epoch))
    ]

    model.fit(x_train, y_train, batch_size=128, epochs=200, callbacks=callbacks)
    _, accuracy = model.evaluate(x_test, y_test)

    plt.imshow(x_train[:1].reshape(32, 32, 3), cmap="gray")
    

    attention = VisualTransformer.attention_map(model, x_train[0])

    plt.matshow(attention.reshape(32, 32), cmap="viridis")
    plt.show()

    # save
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")


    model.save("vit_cifar100")


if __name__ == "__main__":
    run_experiment()
