#!/usr/bin/env python
from __future__ import print_function, unicode_literals

import os
from argparse import ArgumentParser
from glob import glob
from os.path import join as pathjoin

import numpy as np
from scipy.misc import imshow, imread

from hebnet import get_model
from hebnet.utils.dataset import load_prepare_image
from hebnet.utils.settings import idx2char
from utils.settings import HEBNET_LOGO


def load_images(images_dir):
    if os.path.isdir(images_dir):
        fnames = glob(pathjoin(images_dir, '*.jpg')) + \
             glob(pathjoin(images_dir, '*.png'))
    else:
        fnames = [images_dir]
    print('found %d images' % len(fnames))

    images = list()
    for fname in fnames:
        print('reading %s image' % fname)
        img = load_prepare_image(fname)
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        images.append((fname, img, ))

    return images


def predict(model, images):
    for fname, img in sorted(images):
        char_probs = model.predict(img)[0]
        predicted_idx = np.argmax(char_probs)
        predicted_char = idx2char[predicted_idx]

        # for each filname save the image array, the classified char and
        # all char probabilities
        yield (fname, img.reshape(img.shape[2:]), predicted_char, char_probs, )


def k_argmax(char_probs, k):
    """returns the k highest probabilities chars in a list of (char, prob)"""
    max_probs = char_probs.argsort()[-k:][::-1]
    return [(idx2char[idx], char_probs[idx]) for idx in max_probs]


def hebnet_predict(images_dir, model, net_type, show_images=False):
    images = load_images(images_dir)
    model = get_model(model, net_type)

    print('predicting...')
    for fname, img, best_char, char_probs in predict(model, images):
        top5 = k_argmax(char_probs, k=5)

        print('%s best: %s.' % (fname, best_char))
        print('all predictions: %s' % u', '.join(
                '%s:%.4f' % prob for prob in top5
            )
        )
        if show_images:
            imshow(imread(fname, flatten=True))
            imshow(img)


def main():
    parser = ArgumentParser('HebNet - deep neural network for classifying '
                            'hebrew characters and digits.\n'
                            'Authors: Shmuel Amar, Maor Sisso\n')
    print(HEBNET_LOGO)
    parser.add_argument('-d', '--images-dir', required=True)
    parser.add_argument('-m', '--model', default=None)
    parser.add_argument('-t', '--net-type', default=None)
    parser.add_argument('-s', '--show-image', default=False,
                        action='store_true')

    args = parser.parse_args()

    return hebnet_predict(args.images_dir, args.model, args.net_type,
                          args.show_image)


if __name__ == '__main__':
    main()
