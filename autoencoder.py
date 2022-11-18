import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D
from keras.regularizers import L2

def encoder_conv(input_img):
    x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=L2(l2=0.001), padding='same')(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', kernel_regularizer=L2(l2=0.001), padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, (3, 3), activation='relu', kernel_regularizer=L2(l2=0.001), padding='same')(x)

    encoded = Conv2D(256, (3, 3), activation='relu', kernel_regularizer=L2(l2=0.001), padding='same')(x) 
    return encoded

def decoder_conv(encoder_shape):
    x = Conv2D(256, (3, 3), activation='relu', kernel_regularizer=L2(l2=0.001), padding='same')(encoder_shape)

    x = Conv2D(128, (3, 3), activation='relu', kernel_regularizer=L2(l2=0.001), padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2,2))(x)

    x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=L2(l2=0.001), padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2,2))(x)
    decoded = Conv2D(1, (3, 3), activation='tanh', kernel_regularizer=L2(l2=0.001), padding='same')(x)
    return decoded

def display_autoencoder(autoencoder, X_val):
    offset=400

    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(X_val[i+offset,:,:, -1], cmap='gray')
    plt.show()

    for i in range(9):
        plt.subplot(330 + 1 + i)
        output = autoencoder.predict(np.array([X_val[i+offset]]))
        op_image = np.reshape(output[0]*255, (28, 28))
        plt.imshow(op_image, cmap='gray')
    plt.show()
