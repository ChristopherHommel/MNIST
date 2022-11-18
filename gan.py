import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Dense, Input, LeakyReLU, Conv2DTranspose, Flatten
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam

def generator(latent_dim):
    model = tf.keras.models.Sequential()
    input_image = 128 * 7 * 7
    model.add(Dense(input_image, input_dim=latent_dim))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Reshape((7, 7, 128)))

    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    
    model.add(Conv2D(1, (7,7), activation='tanh', padding='same'))
    return model

def discriminator(input_image=(28,28,1)):
    model = Sequential()
    model.add(Conv2D(128, (3,3), strides=(2, 2), padding='same', input_shape=input_image))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    
    model.add(Conv2D(128, (3,3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    return model

def gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001))
    return model

def generate_real_samples(X_train, batch_size):
    ix = randint(0, X_train.shape[0], batch_size)
    X = X_train[ix]
    y = ones((batch_size, 1))
    return X, y

def generate_latent_points(latent_dim, batch_size):
    x_input = randn(latent_dim * batch_size)
    x_input = x_input.reshape(batch_size, latent_dim)
    return x_input
 
def generate_fake_samples(generator, latent_dim, batch_size):
    x_input = generate_latent_points(latent_dim, batch_size)
    X = generator.predict(x_input)
    y = zeros((batch_size, 1))
    return X, y

def show_plot(example,n):
    for i in range(n*n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(example[i, :, :, 0], cmap='gray_r')
    
    plt.show()
