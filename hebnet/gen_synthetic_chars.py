"""
fonts from:
http://www.webjunkie.co.il/ShowFontCat.asp?ID=1
http://www.imarketer.co.il/designing/free-hebrew-fonts
"""
from __future__ import unicode_literals

from glob import glob
from os.path import join as pathjoin

import numpy as np
from scipy.misc import imsave

from hebnet.utils import settings
from hebnet.utils.captcha_image import captcha_image
from hebnet.utils.elastic_distortion import elastic_transform
from hebnet.utils.settings import HEBCHARS, HEBCHARS_DFUS_KTIV_SIMILAR


def PIL2array(img):
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0], 3)


def get_charset(fontpath, all_chars=False):
    if all_chars:
        return HEBCHARS

    folder = fontpath.rsplit('/', 2)[-2]
    if folder.startswith('ktiv'):
        return HEBCHARS
    elif folder.startswith('dfus'):
        return HEBCHARS_DFUS_KTIV_SIMILAR


def get_fonts_and_charset(ktiv_only=False, all_chars=False):
    fonts = list()

    for fontpath in glob(settings.FONTS_DIR_PATTERN):
        charset = get_charset(fontpath, all_chars)
        if not ktiv_only or charset == HEBCHARS:
            fonts.append((fontpath, charset, ))
    return fonts


def get_letter_image(char, fontpath, elastic_trans=True):
    img = captcha_image(char, fontpath)
    img = PIL2array(img)
    if elastic_trans:
        img = elastic_transform(img, kernel_dim=15, alpha=5.5, sigma=35)
    return img


def create_samples(num_samples, outdir, ktiv_only=False, all_chars=False,
                   elastic_trans=True):
    for fontpath, charset in get_fonts_and_charset(ktiv_only, all_chars):
        for char in charset:
            for i in xrange(num_samples):
                img = get_letter_image(char, fontpath, elastic_trans)
                font = fontpath.rsplit('/')[-1]
                img_fname = pathjoin(outdir, u'%s_%s_%d.jpg' % (char, font, i))
                print('saving image %s' % img_fname)
                imsave(img_fname, img)


if __name__ == '__main__':
    create_samples(200, '%s/sample_images' % settings.DATASET_DIR,
                   ktiv_only=True, elastic_trans=True)
