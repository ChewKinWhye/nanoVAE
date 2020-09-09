import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda
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
    x = layers.Dense(512, activation="relu")(encoder_inputs)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_mean_var = layers.Concatenate()([z_mean, z_log_var])
    z_mean_var = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(z_mean_var)
    z_mean = Lambda(lambda x: x[:, 0:latent_dim])(z_mean_var)
    z_log_var = Lambda(lambda x: x[:, latent_dim:])(z_mean_var)

    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    # Build decoder
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(256, activation="relu")(latent_inputs)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(512, activation="relu")(x)
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


def load_vae_predictor(latent_dim):
    predictor_input = keras.Input(shape=(latent_dim * 2,))
    x = layers.Dense(8, activation="relu", kernel_initializer='random_normal',
                     bias_initializer='zeros')(predictor_input)
    x = layers.Dense(8, activation="relu", kernel_initializer='random_normal', 
                     bias_initializer='zeros')(x)
    predictor_output = layers.Dense(1, activation="sigmoid", kernel_initializer='random_normal',
                                    bias_initializer='zeros')(x)
    predictor = keras.Model(predictor_input, predictor_output, name="predictor")
    predictor.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam())
    return predictor
