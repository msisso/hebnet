"""this file is a refactor of https://github.com/mbi/django-simple-captcha
for generating distorted characters from fonts.
"""
import random
import re

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    import Image
    import ImageDraw
    import ImageFont


CAPTCHA_FONT_PATH = 'Vera.ttf'
CAPTCHA_FONT_SIZE = 25
CAPTCHA_LETTER_ROTATION = (-5, 5)
CAPTCHA_BACKGROUND_COLOR = '#ffffff'
CAPTCHA_FOREGROUND_COLOR = '#001100'
CAPTCHA_CHALLENGE_FUNCT = 'captcha.helpers.random_char_challenge'
CAPTCHA_NOISE_FUNCTIONS = ['captcha.helpers.noise_arcs',
                           'captcha.helpers.noise_dots']
CAPTCHA_FILTER_FUNCTIONS = 'captcha.helpers.post_smooth'
CAPTCHA_WORDS_DICTIONARY = '/usr/share/dict/words'
CAPTCHA_PUNCTUATION = ''
CAPTCHA_FLITE_PATH = None
CAPTCHA_TIMEOUT = 5  # Minutes
CAPTCHA_LENGTH = 4  # Chars
CAPTCHA_DICTIONARY_MIN_LENGTH = 0
CAPTCHA_DICTIONARY_MAX_LENGTH = 99
CAPTCHA_IMAGE_SIZE = (28, 28)
CAPTCHA_IMAGE_TEMPLATE = 'captcha/image.html'
CAPTCHA_HIDDEN_FIELD_TEMPLATE = 'captcha/hidden_field.html'
CAPTCHA_TEXT_FIELD_TEMPLATE = 'captcha/text_field.html'
CAPTCHA_FIELD_TEMPLATE = 'captcha/field.html'
CAPTCHA_OUTPUT_FORMAT = None


NON_DIGITS_RX = re.compile(r'[^\d]')
from_top = 4


def getsize(font, text):
    if hasattr(font, 'getoffset'):
        return [x + y for x, y in
                zip(font.getsize(text), font.getoffset(text))]
    else:
        return font.getsize(text)


def makeimg(size):
    if CAPTCHA_BACKGROUND_COLOR == "transparent":
        image = Image.new('RGBA', size)
    else:
        image = Image.new('RGB', size, CAPTCHA_BACKGROUND_COLOR)
    return image


def captcha_image(text, fontpath, scale=1, size=(32, 32)):
    if fontpath.lower().strip().endswith('ttf'):
        font = ImageFont.truetype(fontpath, CAPTCHA_FONT_SIZE * scale)
    else:
        font = ImageFont.load(fontpath)

    image = makeimg(size)

    try:
        PIL_VERSION = int(NON_DIGITS_RX.sub('', Image.VERSION))
    except:
        PIL_VERSION = 116
    xpos = 2

    charlist = []
    for char in text:
        if char in CAPTCHA_PUNCTUATION and len(charlist) >= 1:
            charlist[-1] += char
        else:
            charlist.append(char)
    for char in charlist:
        fgimage = Image.new('RGB', size, CAPTCHA_FOREGROUND_COLOR)
        charimage = Image.new('L', getsize(font, ' %s ' % char), '#000000')
        chardraw = ImageDraw.Draw(charimage)
        chardraw.text((0, 0), ' %s ' % char, font=font, fill='#ffffff')
        if CAPTCHA_LETTER_ROTATION:
            if PIL_VERSION >= 116:
                charimage = charimage.rotate(
                    random.randrange(*CAPTCHA_LETTER_ROTATION),
                    expand=0, resample=Image.BICUBIC)
            else:
                charimage = charimage.rotate(
                    random.randrange(*CAPTCHA_LETTER_ROTATION),
                    resample=Image.BICUBIC
                )
        charimage = charimage.crop(charimage.getbbox())
        maskimage = Image.new('L', size)

        mask = (
            xpos, from_top, xpos + charimage.size[0],
            from_top + charimage.size[1],
        )
        maskimage.paste(charimage, mask)
        size = maskimage.size
        image = Image.composite(fgimage, image, maskimage)
        xpos = xpos + 2 + charimage.size[0]

    if CAPTCHA_IMAGE_SIZE:
        # centering captcha on the image
        mask = (
            int((size[0] - xpos) / 2),
            int((size[1] - charimage.size[1]) / 2 - from_top)
        )
        tmpimg = makeimg(size)
        tmpimg.paste(image, mask)
        image = tmpimg.crop((0, 0, size[0], size[1]))
    else:
        image = image.crop((0, 0, xpos + 1, size[1]))
    return image
