import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse
from tensorflow.keras.optimizers import Adam


def inception_module(layer_in):
    conv1 = layers.Conv2D(32, (1,1), padding='same', activation='relu')(layer_in)
    conv3 = layers.Conv2D(32, (1,3), padding='same', activation='relu')(layer_in)
    conv5 = layers.Conv2D(32, (1,5), padding='same', activation='relu')(layer_in)
    pool = layers.MaxPooling2D((1,3), strides=(1,1), padding='same')(layer_in)
    layer_out = layers.concatenate([conv1, conv3, conv5, pool])
    return layer_out

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def load_vae_dna_model(latent_dim, rc_loss_scale, vae_lr):
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
    encoder = keras.Model(encoder_inputs, z_mean_var, name="encoder")

    # Build decoder
    latent_inputs = keras.Input(shape=(latent_dim*2,))
    x_mean = Lambda(lambda x: x[:, 0:latent_dim])(latent_inputs)
    x_log_var = Lambda(lambda x: x[:, latent_dim:])(latent_inputs)
    x = Lambda(sampling, output_shape=(latent_dim,), name='z')([x_mean, x_log_var])        
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(479, activation="linear")(x)
    decoder_outputs = layers.Reshape((479,))(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    outputs = decoder(encoder(encoder_inputs))
    vae = keras.Model(encoder_inputs, outputs, name='vae_mlp')
    reconstruction_loss = mse(encoder_inputs, outputs)
    reconstruction_loss *= rc_loss_scale
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer=Adam(learning_rate=vae_lr, clipnorm=1.0))
    return encoder, decoder, vae


def load_vae_dna_model_deepsignal(latent_dim, rc_loss_scale, vae_lr):
    # Build encoder
    encoder_inputs = keras.Input(shape=(479,))
    base = Lambda(lambda y: y[:, 0:68])(encoder_inputs)
    base = layers.Reshape((17, 4))(base)
    features = Lambda(lambda y: y[:, 68:119])(encoder_inputs)
    features = layers.Reshape((3, 17))(features)
    features = layers.Permute((2,1))(features)
    top_module = layers.Concatenate(axis=-1)([base, features])
    x = layers.Bidirectional(layers.LSTM(50, return_sequences=True))(top_module)
    top_out = layers.Bidirectional(layers.LSTM(50))(x)
    bottom_module = Lambda(lambda y: y[:, 119:])(encoder_inputs)
    x = layers.Reshape((1, 360, 1))(bottom_module)
    x = layers.Conv2D(filters=64, kernel_size=(1, 7), strides=2)(x)
    # Add in inception layers
    x = layers.MaxPooling2D(pool_size=(1, 3), strides=2)(x)
    x = layers.Conv2D(filters=128, kernel_size=(1, 1), strides=1)(x)
    x = inception_module(x)
    x = layers.Conv2D(filters=32, kernel_size=(1, 7), strides=5)(x)
    bottom_out = layers.Reshape((544,))(x)
    # Classification module which combines top and bottom outputs using FFNN
    x = layers.Concatenate(axis=-1)([top_out, bottom_out])
    x = layers.Dense(256, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_mean_var = layers.Concatenate()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, z_mean_var, name="encoder")

    # Build decoder
    latent_inputs = keras.Input(shape=(latent_dim*2,))
    x_mean = Lambda(lambda y: y[:, 0:latent_dim])(latent_inputs)
    x_log_var = Lambda(lambda y: y[:, latent_dim:])(latent_inputs)
    latent_sampled = Lambda(sampling, output_shape=(latent_dim,), name='z')([x_mean, x_log_var])
    fc_out = layers.Dense(256, activation='relu')(latent_sampled)
    top_module = Lambda(lambda y: y[:, 0:119])(fc_out)
    x = layers.Reshape((17, 7))(top_module)
    x = layers.Bidirectional(layers.LSTM(50, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(50, return_sequences=True))(x)
    x = layers.LSTM(7, return_sequences=True)(x)
    x = layers.Reshape((119,))(x)
    top_out = layers.Dense(119, activation="relu")(x)
    bottom_module = Lambda(lambda x: x[:, 119:])(fc_out)
    x = layers.Reshape((1, 137, 1))(bottom_module)
    x = layers.Conv2D(filters=64, kernel_size=(1, 7), strides=2)(x)
    x = layers.MaxPooling2D(pool_size=(1, 3), strides=2)(x)
    x = layers.Conv2D(filters=128, kernel_size=(1, 1), strides=1)(x)
    x = inception_module(x)
    x = layers.Conv2D(filters=32, kernel_size=(1, 7), strides=5)(x)
    x = layers.Reshape((192,))(x)
    bottom_out = layers.Dense(360, activation="relu")(x)
    decoder_outputs = layers.Concatenate(axis=1)([top_out, bottom_out])
    decoder_outputs = layers.Reshape((479,))(decoder_outputs)
    decoder_outputs = layers.Dense(479)(decoder_outputs)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    outputs = decoder(encoder(encoder_inputs))
    vae = keras.Model(encoder_inputs, outputs, name='vae_mlp')
    reconstruction_loss = mse(encoder_inputs, outputs)
    reconstruction_loss *= rc_loss_scale
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    z_mean_norm = tf.math.divide(tf.subtract(z_mean, tf.reduce_min(z_mean, axis=0)), tf.subtract(tf.reduce_max(z_mean, axis=0), tf.reduce_min(z_mean, axis=0)))
    pairwise_distances_squared = tf.math.add(
        tf.math.reduce_sum(tf.math.square(z_mean_norm), axis=[1], keepdims=True),
        tf.math.reduce_sum(
            tf.math.square(tf.transpose(z_mean_norm)), axis=[0], keepdims=True
        ),
    ) - 2.0 * tf.matmul(z_mean_norm, tf.transpose(z_mean_norm))
    pairwise_distances_squared = tf.reshape(pairwise_distances_squared, [-1])
    labels = encoder_inputs[:, 24:48]
    label_mask = tf.reduce_all(tf.math.equal(tf.expand_dims(labels, axis=0), tf.expand_dims(labels, axis=1)), 2)
    label_mask = tf.math.logical_not(tf.reshape(label_mask, [-1]))
    k_mer_loss = tf.boolean_mask(pairwise_distances_squared, label_mask)
    k_mer_loss = tf.reduce_mean(k_mer_loss) * 0.1
    vae_loss = K.mean(reconstruction_loss + kl_loss - k_mer_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer=Adam(learning_rate=vae_lr, clipnorm=1.0, epsilon=1e-06))
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
