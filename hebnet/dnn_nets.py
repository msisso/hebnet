"""A collection of neural networks templates"""
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D


def build_convnet(nb_classes, img_shape, nb_filters=32, nb_pool=2, nb_conv=3, nb_dense=128, dropout=0.25, activation='relu', nick=''):
    """
    :param nb_filters: number of convolutional filters to use
    :param nb_pool: size of pooling area for max pooling
    :param nb_conv: convolution kernel size
    :param nb_dense: number neurons in fully connected layer
    :param activation: activation function (e.g. relu)
    :return: the compiled model
    """
    net_name = 'convnet_%s_shape-%dX%d_filters-%d_pool-%dX%d_conv-%dX%d_dense-%d_dropout-%.2f_func-%s.model' % (
        nick, img_shape[1], img_shape[2], nb_filters, nb_pool, nb_pool,
        nb_conv, nb_conv, nb_dense, dropout, activation
    )
    model = Sequential()
    model.net_name = net_name

    model.add(Convolution2D(nb_filters, nb_conv,
                            nb_conv, border_mode='valid',
                            input_shape=img_shape))
    model.add(Activation(activation))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(nb_dense))
    model.add(Activation(activation))
    model.add(Dropout(dropout))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


def build_hebnet(nb_classes, img_shape, filter1=32, kernel1=3, filter2=64,
                 kernel2=2, filter3=96, kernel3=2, dense1=512, dense2=384,
                 drop1=0.6, drop2=0.7, func='relu', nick='',
                 ):
    net_name = 'hebnet_%s_shape-%dX%d_f1-%d_k1-%d_f2-%d_k2-%d_f3-%d_k3-%d_' \
               'd1-%.2d-%d_func-%s.model' % (
        nick, img_shape[1], img_shape[2], filter1, kernel1, filter2, kernel2,
        filter3, kernel3, dense1, dense2, func
    )
    model = Sequential()
    model.net_name = net_name

    # conv1
    model.add(Convolution2D(filter1, kernel1, kernel1, input_shape=img_shape,
                            border_mode='same'))
    model.add(Activation(func))
    model.add(Dropout(drop1))

    # conv2 + maxpool1
    model.add(Convolution2D(filter2, kernel2, kernel2))
    model.add(Activation(func))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop1))

    # conv3
    model.add(Convolution2D(filter3, kernel3, kernel3, border_mode='same'))
    model.add(Activation(func))
    model.add(Dropout(drop1))

    # conv4 + maxpool2
    model.add(Convolution2D(filter3, kernel3, kernel3))
    model.add(Activation(func))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop1))

    # fully connected
    model.add(Flatten())
    model.add(Dense(dense1))
    model.add(Activation(func))
    model.add(Dropout(drop2))
    model.add(Dense(dense2))
    model.add(Activation(func))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['acc'])
    return model
