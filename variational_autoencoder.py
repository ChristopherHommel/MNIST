import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, Dense, Input, Dropout, Flatten, Lambda, Reshape, Conv2DTranspose
from scipy.stats import norm

input_img = Input(shape=[28, 28, 1])
latent_space_dim = 2

def display_variational_map(decoder_model):
    size = 30
    digit_size = 28
    figure = np.zeros((digit_size * size, digit_size * size))

    grid_x = norm.ppf(np.linspace(0.01, 0.99, size)) 
    grid_y = norm.ppf(np.linspace(0.01, 0.99, size))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder_model.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()

def display_variational(model, X_val):
    offset=400

    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(X_val[i+offset,:,:, -1], cmap='gray')
    plt.show()


    for i in range(9):
        plt.subplot(330 + 1 + i)
        output = model.predict(np.array([X_val[i+offset]]))
        op_image = np.reshape(output[0]*255, (28, 28))
        plt.imshow(op_image, cmap='gray')
    plt.show()

def sample_latent_features(distribution):
    distribution_mean, distribution_variance = distribution
    batch_size = tf.shape(distribution_variance)[0]
    random = tf.keras.backend.random_normal(shape=(batch_size, tf.shape(distribution_variance)[1]))
    return distribution_mean + tf.exp(0.5 * distribution_variance) * random

#https://github.com/kartikgill/Autoencoders/blob/main/Variational%20Autoencoder.ipynb
#Cell 13
def get_loss(distribution_mean, distribution_variance):
    
    def get_reconstruction_loss(y_true, y_pred):
        reconstruction_loss = tf.keras.losses.mse(y_true, y_pred)
        reconstruction_loss_batch = tf.reduce_mean(reconstruction_loss)
        return reconstruction_loss_batch*28*28
    
    def get_kl_loss(distribution_mean, distribution_variance):
        kl_loss = 1 + distribution_variance - tf.square(distribution_mean) - tf.exp(distribution_variance)
        kl_loss_batch = tf.reduce_mean(kl_loss)
        return kl_loss_batch*(-0.5)
    
    def total_loss(y_true, y_pred):
        reconstruction_loss_batch = get_reconstruction_loss(y_true, y_pred)
        kl_loss_batch = get_kl_loss(distribution_mean, distribution_variance)
        return reconstruction_loss_batch + kl_loss_batch
    
    return total_loss

def variational_autoencoder(input_img):
    x = Conv2D(256, (5,5), activation='relu')(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(128, (3,3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(128, (3,3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    x = Flatten()(x)
    encoder = Dense(128)(x)

    mean = Dense(latent_space_dim, name='mean')(encoder)
    variance = Dense(latent_space_dim, name='log_variance')(encoder)
    latent_encoding = Lambda(sample_latent_features)([mean, variance])

    encoder_model = Model(input_img, latent_encoding)

    decoder_input = Input(shape=(latent_space_dim,))
    y = Dense(128)(decoder_input)
    y = Reshape((1, 1, 128))(y)
    y = Conv2DTranspose(128, (3,3), activation='relu')(y)
    y = BatchNormalization()(y)

    y = Conv2DTranspose(256, (3,3), activation='relu')(y)
    y = BatchNormalization()(y)
    y = UpSampling2D((2,2))(y)

    y = Conv2DTranspose(256, (3,3), activation='relu')(y)
    y = BatchNormalization()(y)
    y = UpSampling2D((2,2))(y)

    decoder_output = Conv2DTranspose(1, (5,5), activation='relu')(y)

    decoder_model = Model(decoder_input, decoder_output)

    encoded = encoder_model(input_img)
    decoded = decoder_model(encoded)

    autoencoder = Model(input_img, decoded)

    return mean, variance, autoencoder, decoder_model, encoder_model
