# coding=utf-8
import math
import os
import random
import shutil
from collections import defaultdict
from os.path import dirname, abspath, join as pathjoin

import cv2
import numpy as np
from scipy.misc import imresize, imread, imsave

from hebnet.utils import imageutils
from hebnet.utils.elastic_distortion import elastic_transform
from hebnet.utils.imageutils import pad_image
from hebnet.utils.settings import idx2char, RAND_SEED, char2idx

here = dirname(abspath(__file__))
ALL_IMAGES_DIR = pathjoin(here, '../../dataset/all_images')
HUMAN_DATASET_DIR = pathjoin(here, '../../dataset/human')
MIN_HEIGHT = 1
MIN_WIDTH = 5


# preprocessing image functions
def normalize_image(img):
    """normalize image between 0-1"""
    if img.dtype != np.float32:
        img = img.astype('float32')

    if (img > 1).any():
        img /= 255.0
    return img


def image_boxing(img, min_height=MIN_HEIGHT, min_width=MIN_WIDTH):
    a = np.where(img != 0)
    bottom = np.min(a[0])
    top = np.max(a[0])
    left = np.min(a[1])
    right = np.max(a[1])

    if top - bottom < min_height:
        diff = (min_height - (top - bottom)) / 2.0
        top += diff
        bottom -= diff

    if right - left < min_width:
        diff = (min_width - (right - left)) / 2.0
        right += diff
        left -= diff

    box = img[bottom: top, left: right]
    return box


# TODO: fix this
# TODO: unify imageutils from here
def load_prepare_image(fname, out_size=(32, 32), border_pad=0, normalize=True):
    # read image in grayscale
    img = imread(fname, flatten=True)

    # box image - cut relevant box of the image (i.e. the character)
    img = image_boxing(img)

    # white pad do NxN image
    new_size = (max(img.shape), max(img.shape))
    img = pad_image(img, new_size=new_size, color=255)

    # resize to wanted output size and add white border
    img_size = (out_size[0] - border_pad, out_size[1] - border_pad)
    img = imresize(img, size=img_size)
    # img = pad_image(img, new_size=out_size, color=255)

    # convert to [0-1] numpy array
    if normalize:
        img = normalize_image(img)
    return img


# dataset functions
def make_all_imgs(srcdirs, outdir=ALL_IMAGES_DIR):
    os.makedirs(outdir)
    fnames = imageutils.get_filenames(srcdirs)
    count = 0

    for fname in fnames:
        _, folder, set_id, name = fname.rsplit('/', 3)
        label = imageutils.get_label_from_filename(fname)
        ext = fname.rsplit('.', 1)[1]

        new_name = pathjoin(outdir, u'%s-%s-%s-%d.%s' % (label, set_id, folder, count, ext))

        if os.path.exists(new_name):
            raise ValueError(u'already exists %s' % new_name)

        print new_name, fname

        with open(fname) as src:
            with open(new_name, 'wb') as dst:
                shutil.copyfileobj(src, dst)

        count += 1


def get_fname_label(fname):
    fname = fname.decode('utf8')
    return fname.rsplit('/', 1)[-1][0]


def train_test_split(images_dir=ALL_IMAGES_DIR, valid_ratio=0.15,
                     test_ratio=0.15):
    label2files = defaultdict(list)
    for fname in imageutils.get_filenames([images_dir]):
        label = get_fname_label(fname)
        label2files[label].append(fname)

    if len(label2files) != len(idx2char):
        raise ValueError('not enough classes')

    train = list()
    valid = list()
    test = list()

    random.seed(RAND_SEED)
    for label, fnames in label2files.iteritems():
        n = int(math.ceil(len(fnames) * test_ratio))
        n_v = n + int(math.ceil((len(fnames) - n) * valid_ratio))
        random.shuffle(fnames)
        test.extend(fnames[:n])
        valid.extend(fnames[n:n_v])
        train.extend(fnames[n_v:])

    assert set(train) & set(test) == set()
    assert set(valid) & set(test) == set()
    assert set(train) & set(valid) == set()
    return train, valid, test


# TODO: dont load prepare image here but before training to compare approaches
def make_train_test_dataset(outdir, images_dir=ALL_IMAGES_DIR, test_ratio=0.15,
                            valid_ratio=0.15):
    train_dir = pathjoin(outdir, 'train')
    valid_dir = pathjoin(outdir, 'validation')
    test_dir = pathjoin(outdir, 'test')

    for dirname in [outdir, train_dir, valid_dir, test_dir]:
        if not os.path.exists(dirname):
            os.mkdir(dirname)

    files_train, files_valid, files_test = train_test_split(
        images_dir, test_ratio, valid_ratio
    )

    print('sample count: train/test/validation %d/%d/%d' %
          (len(files_train), len(files_valid), len(files_test)))

    files_dirs = [
        (files_train, train_dir),
        (files_valid, valid_dir),
        (files_test, test_dir),
    ]
    for fnames, dirname in files_dirs:
        for fname in fnames:
            out_fname = fname.rsplit('/', 1)[-1]
            fpath = pathjoin(dirname, out_fname)
            img = imread(fname)
            print('saving image to %s' % fpath)
            imsave(fpath, img)

    # for fname in files_train:
    #     out_fname = fname.rsplit('/', 1)[-1]
    #     img = load_prepare_image(fname)
    #     fpath = pathjoin(train_dir, out_fname)
    #     imsave(fpath, img)
    #
    # for fname in files_test:
    #     out_fname = fname.rsplit('/', 1)[-1]
    #     img = load_prepare_image(fname)
    #     fpath = pathjoin(test_dir, out_fname)
    #     imsave(fpath, img)


def is_not_distorted_fname(fname):
    return '-distort' not in fname


def read_dataset_from_dir(srcdir=HUMAN_DATASET_DIR, fonts=False, distort=False):
    train_files = imageutils.get_filenames(pathjoin(srcdir, 'train'))
    if fonts:
        train_files += imageutils.get_filenames(pathjoin(srcdir, 'train/font_images'))
    if not distort:
        train_files = filter(is_not_distorted_fname, train_files)

    valid_files = imageutils.get_filenames(pathjoin(srcdir, 'validation'))
    test_files = imageutils.get_filenames(pathjoin(srcdir, 'test'))

    random.shuffle(train_files)
    random.shuffle(test_files)

    X_train = list()
    y_train = list()
    X_valid = list()
    y_valid = list()
    X_test = list()
    y_test = list()

    # TODO: refactor ignore labels
    for fname in train_files:
        if get_fname_label(fname) in '0123456789 ': continue
        X_train.append(load_prepare_image(fname))
        y_train.append(char2idx[get_fname_label(fname)])

    for fname in valid_files:
        if get_fname_label(fname) in '0123456789 ': continue
        X_valid.append(load_prepare_image(fname))
        y_valid.append(char2idx[get_fname_label(fname)])

    for fname in test_files:
        if get_fname_label(fname) in '0123456789 ': continue
        X_test.append(load_prepare_image(fname))
        y_test.append(char2idx[get_fname_label(fname)])

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_valid = np.asarray(X_test)
    y_valid = np.asarray(y_test)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    # add filter dimension to the image (image, filters, row, col)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1], X_train.shape[2]))
    X_valid = X_valid.reshape((X_valid.shape[0], 1, X_valid.shape[1], X_valid.shape[2]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1], X_test.shape[2]))
    nb_classes = len(set(y_train) | set(y_test))

    print 'data shapes:'
    print X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape, nb_classes
    return X_train, y_train, X_valid, y_valid, X_test, y_test, nb_classes


def distort_dataset(srcdir=HUMAN_DATASET_DIR, multiply=5):
    train_files = imageutils.get_filenames(pathjoin(srcdir, 'train'))

    for fname in train_files:
        if '-distort' in fname:
            continue
        img = imread(fname, flatten=True)
        for j in xrange(multiply):
            dist_img = elastic_transform(img, kernel_dim=11, sigma=6, alpha=8.5)
            dist_fname = fname.replace('-', '-distort%d-' % j, 1)
            print dist_fname
            imsave(dist_fname, dist_img)


if __name__ == '__main__':
    if raw_input('do you want to create dataset(y|n):') != 'y':
        print('exiting..')
    else:
        make_train_test_dataset('../../dataset/human3')
