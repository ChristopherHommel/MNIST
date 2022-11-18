from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D, BatchNormalization
from keras.regularizers import L2

def conv_model(input_shape, num_classes):
    y = Conv2D(32, (3,3), kernel_regularizer=L2(l2=0.001), padding='same')(input_shape)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = MaxPooling2D(pool_size=(2,2))(y)

    y = Conv2D(64, (3,3), kernel_regularizer=L2(l2=0.001), padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = MaxPooling2D(pool_size=(2,2))(y)

    y = Conv2D(128, (3,3), kernel_regularizer=L2(l2=0.001), padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = MaxPooling2D(pool_size=(2,2))(y)

    y = Conv2D(256, (3,3), kernel_regularizer=L2(l2=0.001), padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = MaxPooling2D(pool_size=(2,2))(y)

    y = Flatten()(y)
    y = Model(inputs=input_shape, outputs=y)

    z = Dense(256, kernel_regularizer=L2(l2=0.001))(y.output)
    z = Activation('relu')(z)
    z = Dropout(0.5)(z)

    z = Dense(128, kernel_regularizer=L2(l2=0.001))(z)
    z = Activation('relu')(z)
    z = Dropout(0.5)(z)

    z = Dense(64, kernel_regularizer=L2(l2=0.001))(z)
    z = Activation('relu')(z)
    z = Dropout(0.5)(z)

    z = Dense(num_classes, activation='sigmoid')(z)
 
    model = Model(inputs=y.input,outputs=z)
    return model
