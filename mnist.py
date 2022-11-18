import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from numpy import ones
from numpy import vstack
from tqdm import tqdm
from keras.utils.vis_utils import plot_model

from convolutional import conv_model 
from autoencoder import encoder_conv, decoder_conv, display_autoencoder
from variational_autoencoder import variational_autoencoder, get_loss, display_variational_map, display_variational
from gan import generator, discriminator, gan, generate_real_samples, generate_latent_points, generate_fake_samples, show_plot

import gc

gc.enable()
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.75, allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()

def load(num_channels=1):
    (X_train, y_train), (X_test, y_test) = load_data()

    splits = 8000
    
    X_val = X_test[:splits]
    y_val = y_test[:splits]
    X_test = X_test[splits:]
    y_test = y_test[splits:]

    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)
    y_test = to_categorical(y_test, 10)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], num_channels))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], X_val.shape[2], num_channels))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], num_channels))

    shapes = {'X_train':X_train.shape,
              'y_train':y_train.shape,
              'X_test':X_test.shape,
              'y_test':y_test.shape,
              'X_val':X_val.shape,
              'y_val':y_val.shape,}

    return X_train, y_train, X_test, y_test, X_val, y_val, shapes

def save_summary(model, filename):
    plot_model(model, to_file=filename, show_shapes=True, show_layer_names=True)


def explore_data():
    (X, y), (_, _) = load_data()

    images = 10
    num_row = 2
    num_col = 5

    fig, axes = plt.subplots(num_row, num_col, figsize=(10,10))
    for i in range(images):
        ax = axes[i//num_col, i%num_col]
        ax.imshow(X[i], cmap='gray')
        ax.set_title(f'Label: {y[i]}')   
    plt.show()

def convolutional():
    num_classes = 10
    early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, restore_best_weights=True, mode='min')
    
    X_train, y_train, X_test, y_test, X_val, y_val, shapes = load()
    
    model = conv_model(Input(shape=X_train[0].shape), num_classes)

    train_datagen = ImageDataGenerator(rotation_range=90,
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        shear_range=0.25,
                                        zoom_range=(0.9, 1.1),
                                        horizontal_flip=True,
                                        vertical_flip=True, 
                                        fill_mode='constant',
                                        cval=0)

    opt = Adam(learning_rate=1e-3,decay=1e-5)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)

    history = model.fit(train_datagen.flow(X_train, y_train, shuffle=True, batch_size=64),
                        epochs=100,
                        validation_data=[X_val,y_val],
                        callbacks=[early_stopper])

    evaluation = model.evaluate(X_test,y_test)
    model.save('./models/convolutional/' + 'Convolutional-{:.4f}'.format(evaluation[1]))
    save_summary(model, './images/convolutional/summary.png') 

def autoencoder():
    X_train, _, X_test, _, X_val, _, _ = load()
    print(X_train.shape, X_test.shape, X_val.shape)
    early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, restore_best_weights=True, mode='min')

    input_img = Input(shape=[28, 28, 1])

    model = Model(inputs=input_img, outputs=decoder_conv(encoder_conv(input_img)))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['accuracy'])

    train_datagen = ImageDataGenerator(rotation_range=90,
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        shear_range=0.25,
                                        zoom_range=(0.9, 1.1),
                                        horizontal_flip=True,
                                        vertical_flip=True, 
                                        fill_mode='constant',
                                        cval=0)

    history = model.fit(train_datagen.flow(X_train, X_train, shuffle=True, batch_size=64), 
                            epochs=100,
                            validation_data=(X_val,X_val),
                            callbacks=[early_stopper])
    
    evaluation = model.evaluate(X_test, X_test)
    model.save('./models/autoencoder/' + 'Autoencoder-{:.4f}'.format(evaluation[1]))
    save_summary(model, './images/autoencoder/summary.png') 
    display_autoencoder(model, X_val)

def variational_autoencoder_conv():
    tf.compat.v1.disable_eager_execution()
    early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, restore_best_weights=True, mode='min')
    
    X_train, _, X_test, _, X_val, _, _ = load()
    input_img = Input(shape=[28, 28, 1])

    mean, variance, model, decoder_model, encoder_model = variational_autoencoder(input_img)
    custom_loss = get_loss(mean, variance)

    model.compile(loss=custom_loss, optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'], experimental_run_tf_function=False)

    model.fit(X_train, X_train,
                    shuffle=True,
                    epochs=100,
                    validation_data=(X_val,X_val),
                    use_multiprocessing=True,
                    workers=4,
                    callbacks=[early_stopper])
    
    evaluation = model.evaluate(X_test, X_test)

    model.save('./models/variational_autoencoder/' + 'Variational_Autoencoder-{:.4f}'.format(evaluation[1]))
    save_summary(model, './images/variational_autoencoder/summary.png')
    save_summary(decoder_model, './images/variational_autoencoder/decoder_summary.png') 
    save_summary(encoder_model, './images/variational_autoencoder/encoder_summary.png')
    display_variational(model, X_test)
    display_variational_map(decoder_model)
    
       

def gan_train(latent_dim=100, epochs=100, batches=256):
    X_train, _, _, _, _, _, _ = load()

    X_train = X_train * 255
    X_train = (X_train - 127.5) / 127.5

    generator_model = generator(latent_dim)
    discriminator_model = discriminator()
    gan_model = gan(generator_model, discriminator_model)

    half_batch = int(batches / 2)
    for j in range(epochs):

        for _ in tqdm(range(int(X_train.shape[0] / batches))):

            X_real, y_real = generate_real_samples(X_train, half_batch)

            X_fake, y_fake = generate_fake_samples(generator_model, latent_dim, half_batch)

            X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))

            d_loss, _ = discriminator_model.train_on_batch(X, y)

            X_gan = generate_latent_points(latent_dim, batches)
            y_gan = ones((batches, 1))
            
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
        
        print('{}/{}'.format(j+1, epochs), 'd_loss:', d_loss, ' g_loss:', g_loss)

    latent_points = generate_latent_points(latent_dim, 25)
    generated_examples = generator_model.predict(latent_points)

    generator_model.save('./models/gan/' + 'GAN-{:.4f}'.format(g_loss))
    save_summary(generator_model, './images/gan/summary_generator.png') 
    save_summary(discriminator_model, './images/gan/summary_discriminator.png') 
    save_summary(gan_model, './images/gan/summary_gan.png') 

    show_plot(generated_examples, 5)

if __name__ == '__main__':
    convolutional()
    autoencoder()
    variational_autoencoder_conv()
    gan_train()

