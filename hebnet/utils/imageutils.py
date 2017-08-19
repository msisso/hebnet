#!/usr/bin/env python
# coding=utf-8
from __future__ import unicode_literals

from glob import glob
from os.path import join as pathjoin

import numpy as np

from hebnet.utils.settings import HEBCHARS, DIGITS


def pad_image(img, new_size=(256, 256), color=1.0):
    padded_img = np.zeros(new_size, dtype=img.dtype)

    y_pad = (new_size[0] - img.shape[0]) // 2
    x_pad = ((new_size[1] - img.shape[1]) // 2) + img.shape[1] % 2
    for i in range(padded_img.shape[0]):
        for j in range(padded_img.shape[1]):
            if y_pad <= i < img.shape[0] + y_pad and \
                    x_pad <= j < img.shape[1] + x_pad:
                padded_img[i, j] = img[i - y_pad, j - x_pad]
            else:
                padded_img[i, j] = color
    return padded_img


def get_label_from_filename(fname, first=False):
    if 'sample_images' in fname or 'outimgs' in fname:
        first = True
    fname = fname.replace('(', '').replace(')', '')
    fname = fname.rsplit('/', 1)[1]

    if first:
        return fname[0]
    if 'outimgs_' in fname:
        fname = fname.split('outimgs_', 1)[1]
    elif '_' in fname:
        fname = fname.rsplit('_', 1)[1]

    if fname[0] == ' ' or fname.startswith('רווח'):
        return ' '
    if fname[0] in HEBCHARS:
        return fname[0]
    if fname[0] in DIGITS:
        return fname[0]

    raise ValueError('unknown filename: "%s"' % fname)


def get_filenames(src_dirs):
    if not isinstance(src_dirs, (list, tuple, set)):
        src_dirs = [src_dirs]

    fnames = []
    for src_dir in src_dirs:
        for ext in ['*.jpg', '*.png']:
            fnames.extend(glob(pathjoin(src_dir, ext)))
    return sorted(set(fnames))
