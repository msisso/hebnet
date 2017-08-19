#!/usr/bin/env python
from argparse import ArgumentParser
from bisect import bisect

import numpy as np
from scipy import ndimage
from sklearn.svm import SVC

from hebnet.utils import mlutils, settings
from hebnet.utils.dataset import read_dataset_from_dir

SOBEL_VER = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1],
])

SOBEL_HOR = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1],
])


def crop_image(img, dim):
    sub_images = list()
    vertical = np.vsplit(img, img.shape[0] / dim[0])

    for vert_arr in vertical:
        for hor_arr in np.hsplit(vert_arr, img.shape[1] / dim[1]):
            sub_images.append(hor_arr)

    return np.array(sub_images)


def calc_img_integral(img):
    hor_arr = ndimage.convolve(img, SOBEL_HOR, mode='constant', cval=0.0)
    ver_arr = ndimage.convolve(img, SOBEL_HOR, mode='constant', cval=0.0)
    # ignore dividing by zero
    np.seterr(divide='ignore')
    # return np.degrees(np.arctan(((hor_arr ** 2) + (ver_arr ** 2)) ** 0.5))
    return np.degrees(np.arctan(ver_arr / hor_arr))


def img_hog(image, crop_dim, deg_bucket_size):
    if len(image.shape) == 3:
        image = image.reshape(image.shape[1:])

    subimg_dim = (image.shape[0] * image.shape[0]) / \
        float(crop_dim[0] * crop_dim[1])

    if not subimg_dim.is_integer():
        raise ValueError('cannot divide image of shape %s to %s subimages' %
                         (image.shape, crop_dim))

    hog_vec_size = (int(subimg_dim), 180 / deg_bucket_size)
    hog_vector = np.zeros(hog_vec_size)

    img_integral = calc_img_integral(image)
    cropped_images = crop_image(img_integral, dim=crop_dim)

    buckets = range(0, 180, deg_bucket_size)
    thresholds = buckets[1:]

    for i, cropped_img in enumerate(cropped_images):
        for _, angle in np.ndenumerate(cropped_img):
            j = bisect(thresholds, angle)
            hog_vector[i, j] += 1

    return hog_vector.reshape(-1)


def get_hog_vectors(data, crop_dim=None, deg_bucket_size=None):
    crop_dim = crop_dim or settings.DEFAULT_CROP_DIM
    if isinstance(crop_dim, (float, int)):
        crop_dim = (crop_dim, crop_dim)
    deg_bucket_size = deg_bucket_size or settings.DEFAULT_DEGREE_BUCKET_SIZE
    lst = []
    for x in data:
        lst.append(img_hog(x, crop_dim, deg_bucket_size))
    return np.array(lst)


def get_svm_classifier(X_train, Y_train, kernel=None, svm_err_cost=None):
    kernel = kernel or settings.DEFAULT_SVM_KERNEL
    svm_err_cost = svm_err_cost or settings.DEFAULT_SVM_ERR_COST
    clf = SVC(kernel=kernel, random_state=settings.RAND_SEED,
              class_weight='balanced', C=svm_err_cost)
    clf.fit(X_train, Y_train)
    return clf


def hog_train_and_predict(X_train, X_test, Y_train, Y_test, crop_dim=None,
                          deg_bucket_size=None, kernel=None, svm_cost=None):
    """train and predict hog with SVM

    :param X_train: training data images
    :param X_test: test data images
    :param Y_train: train labels (classes)
    :param Y_test: test labels (classes)
    :param crop_dim: dimension to crop the image into subimages - (X,X)
    :param deg_bucket_size: degree ranges
    :param kernel: the SVM kernel
    :return: predicted labels
    """
    print('getting hog vectors of train data')
    X_train_hog = get_hog_vectors(X_train, crop_dim, deg_bucket_size)

    print('getting hog vectors of test data')
    X_test_hog = get_hog_vectors(X_test, crop_dim, deg_bucket_size)

    print('training the classifier')
    clf = get_svm_classifier(X_train_hog, Y_train, kernel, svm_cost)

    stats_fname = '%s/hog_crop-%d_bucket-%d_kern-%s-%d.png' % \
        (settings.MODELS_DIR, crop_dim, deg_bucket_size, kernel, svm_cost)
    return mlutils.predict_class(
            clf, X_test_hog, Y_test, labels=settings.idx2char,
            stats_fname=stats_fname
    )


def main(str_args=None):
    parser = ArgumentParser()
    parser.add_argument('--sample-ratio', default=None, type=float)
    parser.add_argument('--crop-dim', default=None, type=int)
    parser.add_argument('--degree-bucket-size', default=None, type=int)
    parser.add_argument('--kernel', default=None, type=str)
    parser.add_argument('--svm-cost', default=None, type=float)

    args = parser.parse_args(str_args)

    X_train, Y_train, X_test, Y_test, nb_classes = read_dataset_from_dir()

    return hog_train_and_predict(
        X_train,
        X_test,
        Y_train,
        Y_test,
        crop_dim=args.crop_dim,
        deg_bucket_size=args.degree_bucket_size,
        kernel=args.kernel,
        svm_cost=args.svm_cost,
    )


if __name__ == '__main__':
    main()
