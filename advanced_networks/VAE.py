import tensorflow as tf


class VariationalAutoencoder(tf.keras.Model):
    def __init__(self, latent_dim, encoder, decoder):
        super(VariationalAutoencoder, self).__init__()

        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                *encoder,
                tf.keras.layers.Dense(latent_dim * 2, name='latent_space'),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                *decoder,
            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        eps = tf.random.normal(shape=(batch, dim))
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def call(self, inputs):
        mean, logvar = self.encode(inputs)
        z = self.reparameterize(mean, logvar)
        return self.decode(z)
    
    def get_encoder(self):
        latent_output = self.encoder.get_layer('latent_space').output # 256 - 128 for mean and 128 for logvar
        encoder = tf.keras.Model(inputs=self.encoder.input, outputs=latent_output[:, :self.latent_dim], name='encoder')
        return encoder