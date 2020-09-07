import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def load_vae_dna_model(latent_dim, rc_loss_scale):
    # Build encoder
    encoder_inputs = keras.Input(shape=(479,))
    x = layers.Dense(256, activation="relu")(encoder_inputs)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = keras.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    # Build decoder
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(32, activation="relu")(latent_inputs)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(479, activation="linear")(x)
    decoder_outputs = layers.Reshape((479,))(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    outputs = decoder(encoder(encoder_inputs)[2])
    vae = keras.Model(encoder_inputs, outputs, name='vae_mlp')

    reconstruction_loss = mse(encoder_inputs, outputs)
    reconstruction_loss *= rc_loss_scale
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    return encoder, decoder, vae


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VaeDNA(keras.Model):
    def __init__(self, latent_dim, rc_loss_scale, **kwargs):
        super(VaeDNA, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.encoder = self.load_encoder()
        self.decoder = self.load_decoder()
        self.rc_loss_scale = rc_loss_scale

    def load_encoder(self):
        encoder_inputs = keras.Input(shape=(479,))
        x = layers.Dense(256, activation="relu")(encoder_inputs)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dense(32, activation="relu")(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        #encoder.summary()
        return encoder

    def load_decoder(self):
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(32, activation="relu")(latent_inputs)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dense(479, activation="linear")(x)
        decoder_outputs = layers.Reshape((479,))(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        #decoder.summary()
        return decoder

    def train_step(self, data):
        # if isinstance(data, tuple):
        #     data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            mse = keras.losses.MeanSquaredError()
            reconstruction_loss = tf.reduce_mean(
                mse(data, reconstruction)
            )
            reconstruction_loss *= self.rc_loss_scale
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

class VaeRNA(keras.Model):
    def __init__(self, latent_dim, input_dim, rc_loss_scale, **kwargs):
        super(VaeRNA, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.rc_loss_scale = rc_loss_scale
        self.encoder = self.load_encoder()
        self.decoder = self.load_decoder()

    def load_encoder(self):
        encoder_inputs = keras.Input(shape=(self.input_dim,))
        x = layers.Dense(8, activation="relu", kernel_initializer='random_normal',
                         bias_initializer='zeros')(encoder_inputs)
        x = layers.Dense(8, activation="relu", kernel_initializer='random_normal',
                         bias_initializer='zeros')(x)
        x = layers.Dense(4, activation="relu", kernel_initializer='random_normal',
                         bias_initializer='zeros')(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        #encoder.summary()
        return encoder

    def load_decoder(self):
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(4, activation="relu", kernel_initializer='random_normal',
                         bias_initializer='zeros')(latent_inputs)
        x = layers.Dense(8, activation="relu", kernel_initializer='random_normal',
                         bias_initializer='zeros')(x)
        x = layers.Dense(self.input_dim, activation="linear", kernel_initializer='random_normal',
                         bias_initializer='zeros')(x)
        decoder_outputs = layers.Reshape((self.input_dim,))(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        #decoder.summary()
        return decoder

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            mse = keras.losses.MeanSquaredError()
            reconstruction_loss = tf.reduce_mean(mse(data, reconstruction))

            reconstruction_loss *= self.rc_loss_scale
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


def load_vae_predictor(latent_dim):
    predictor_input = keras.Input(shape=(latent_dim * 2,))
    x = layers.Dense(8, activation="relu", kernel_initializer='random_normal',
                     bias_initializer='zeros')(predictor_input)
    x = layers.Dense(16, activation="relu", kernel_initializer='random_normal', 
                     bias_initializer='zeros')(x)
    x = layers.Dense(8, activation="relu", kernel_initializer='random_normal', 
                     bias_initializer='zeros')(x)
    predictor_output = layers.Dense(1, activation="sigmoid", kernel_initializer='random_normal',
                                    bias_initializer='zeros')(x)
    predictor = keras.Model(predictor_input, predictor_output, name="predictor")
    predictor.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(), 
            metrics=[tf.keras.metrics.Accuracy()])
    #predictor.summary()
    return predictor
