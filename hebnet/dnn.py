import gc
from argparse import ArgumentParser
from os.path import join as pathjoin

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from hebnet import dnn_nets
from hebnet.utils import mlutils, settings
from hebnet.utils.dataset import read_dataset_from_dir
from hebnet.utils.elastic_distortion import elastic_transform

np.random.seed(settings.RAND_SEED)  # for reproducibility


def train_dnn(model, X_train, Y_train, X_test, Y_test, batch_size=128, nb_epoch=12):
    """
    :param model: the nueron netrwok that we built with the function above
    :param X_train: pixels array of all the train hebnet data
    :param Y_train: labels of the train images
    :param X_test: pixels array of all the test hebnet data
    :param Y_test: labels of the test images
    :param name: the name that will be saved of the nueron network after train
    :param batch_size: number of iterating images of train data
    :param nb_epoch: number of training epochs
    :return: the trained model
    """
    print('training network name %s' % model.net_name)
    # convert class vectors to binary class matrices
    for epoch in xrange(nb_epoch):
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1,
                  verbose=1, validation_data=(X_test, Y_test))

        fname = pathjoin(settings.MODELS_DIR, 'new-epoch%d-%s' % (epoch + 1, model.net_name))
        print('saving model to: %s' % fname)
        model.save_weights(fname, overwrite=True)
    return model


class ExtendedImageDataGenerator(ImageDataGenerator):
    def __init__(self, elastic_distortion=False, dist_kdim=13, dist_sigma=6,
                 dist_alpha=5., *a, **kw):
        super(ExtendedImageDataGenerator, self).__init__(*a, **kw)
        self.elastic_distortion = elastic_distortion
        self.dist_kdim = dist_kdim
        self.dist_sigma = dist_sigma
        self.dist_alpha = dist_alpha

    def random_transform(self, x):
        if self.elastic_distortion:
            x *= 255.
            x = elastic_transform(x.reshape(x.shape[1:]),
                                  kernel_dim=self.dist_kdim,
                                  sigma=self.dist_sigma, alpha=self.dist_alpha)
            x = x.reshape((1, ) + x.shape)
            x /= 255.
        return super(ExtendedImageDataGenerator, self).random_transform(x)


def gen_train_dnn(model, X_train, Y_train, X_valid, Y_valid, batch_size=128, nb_epoch=12):
    """
    :param model: the nueron netrwok that we built with the function above
    :param X_train: pixels array of all the train hebnet data
    :param Y_train: labels of the train images
    :param X_valid: pixels array of all the test hebnet data
    :param Y_valid: labels of the test images
    :param name: the name that will be saved of the nueron network after train
    :param batch_size: number of iterating images of train data
    :param nb_epoch: number of training epochs
    :return: the trained model
    """
    seed = 42

    # TODO: check zca whitening
    # TODO: because this is slow we can enlarge model (GPU is idle)
    train_gen = ExtendedImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        # zca_whitening=True,
        shear_range=0.5,
        elastic_distortion=True,
        dist_alpha=6.5,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode='constant',
        zoom_range=[0.7, 1.1],
        cval=1.0,
    )

    train_gen.fit(X_train, seed=seed)
    train_flow = train_gen.flow(
        X_train, Y_train, batch_size=batch_size, seed=seed
    )

    # fits the model on batches with real-time data augmentation:
    print('training network name %s' % model.net_name)
    # convert class vectors to binary class matrices
    for epoch in xrange(nb_epoch):
        model.fit_generator(
            generator=train_flow,
            validation_data=(X_valid, Y_valid),
            samples_per_epoch=((len(X_train) // batch_size) * batch_size) * 10,
            nb_epoch=1,
            verbose=1,
            nb_worker=8,
            pickle_safe=True
        )

        fname = pathjoin(settings.MODELS_DIR, 'new3-epoch%d-%s' % (epoch + 1, model.net_name))
        print('saving model to: %s' % fname)
        model.save_weights(fname, overwrite=True)
        gc.collect()  # memory leak bug

    return model


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-p', '--predict', action='store_true')
    parser.add_argument('-l', '--load-model', default=None)
    parser.add_argument('-e', '--epochs', default=200, type=int)

    return parser.parse_args()


def main():
    args = parse_args()

    print('loading data')
    X_train, y_train, X_valid, y_valid, X_test, y_test, nb_classes = read_dataset_from_dir(pathjoin(settings.DATASET_DIR, 'human3'))  # fonts=True, distort=True)
    Y_train, Y_valid, Y_test = mlutils.num2categorical(y_train, y_valid, y_test, nb_classes)

    print('building model')
    model = dnn_nets.build_hebnet(nb_classes, img_shape=X_train[0].shape)

    if args.load_model:
        print('loading weights from %s' % args.load_model)
        model.load_weights(args.load_model)

    if args.train:
        print('training model')
        model = gen_train_dnn(model, X_train, Y_train, X_valid, Y_valid, batch_size=32, nb_epoch=args.epochs)

    if args.predict:
        print('predicting model')
        return mlutils.predict_class(
                model, X_test, Y_test, labels=settings.idx2char,
                stats_fname='%s/%s' % (settings.MODELS_DIR, model.net_name)
        )


if __name__ == '__main__':
    main()
