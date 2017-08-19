from __future__ import print_function, unicode_literals

import numpy as np
import six
from scipy.misc import imread, imresize

from hebnet import dnn_nets
from hebnet.utils.dataset import image_boxing, normalize_image
from hebnet.utils.imageutils import pad_image
from hebnet.utils.settings import N_CLASSES, idx2char, BEST_MODEL, \
    BEST_MODEL_TYPE

MODEL = None


def get_model(model_file=None, net_type=None, nb_classes=N_CLASSES,
              size=(1, 32, 32)):
    global MODEL

    model_file = model_file or BEST_MODEL
    net_type = net_type or BEST_MODEL_TYPE

    if MODEL is None:
        print('loading model from %s of type %s...' % (model_file, net_type))
        try:
            model_class = getattr(dnn_nets, 'build_%s' % net_type)
        except AttributeError:
            raise ValueError('net of type %s not exists' % net_type)

        MODEL = model_class(nb_classes, size)
        MODEL.load_weights(model_file)
        print('done')
    return MODEL


def prepare_image(fname_or_img, out_size=(32, 32), border_pad=0):
    # read image in grayscale
    if isinstance(fname_or_img, six.string_types):
        img = imread(fname_or_img, flatten=True)
    else:
        img = fname_or_img

    # box image - cut relevant box of the image (i.e. the character)
    img = image_boxing(img)

    # white pad do NxN image
    new_size = (max(img.shape), max(img.shape))
    img = pad_image(img, new_size=new_size, color=255)

    # resize to wanted output size and add white border
    img_size = (out_size[0] - border_pad, out_size[1] - border_pad)
    img = imresize(img, size=img_size)
    img = pad_image(img, new_size=out_size, color=255)

    # convert to [0-1] numpy array
    img = normalize_image(img)
    img = img.reshape(1, 1, img.shape[0], img.shape[1])
    return img


def predict(fname_or_img):
    model = get_model()
    prep_img = prepare_image(fname_or_img)
    all_chars_probs = model.predict(prep_img)[0]
    predicted_idx = np.argmax(all_chars_probs)
    predicted_char = idx2char[predicted_idx]
    return predicted_char, all_chars_probs
