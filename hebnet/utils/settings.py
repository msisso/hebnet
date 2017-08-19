# coding=utf-8
from os.path import pardir, dirname, abspath, join as pathjoin
here = abspath(dirname(__file__))

HEBNET_LOGO = '''
 _   _      _     _   _      _
| | | |    | |   | \ | |    | |
| |_| | ___| |__ |  \| | ___| |_
|  _  |/ _ \ '_ \| . ` |/ _ \ __|
| | | |  __/ |_) | |\  |  __/ |_
\_| |_/\___|_.__/\_| \_/\___|\__|

'''

DATASET_DIR = pathjoin(here, pardir, pardir, 'dataset')
MODELS_DIR = pathjoin(here, pardir, 'models')
RAND_SEED = 1337

DEFAULT_CROP_DIM = (4, 4)
DEFAULT_DEGREE_BUCKET_SIZE = 30
DEFAULT_SVM_KERNEL = 'rbf'
DEFAULT_SVM_ERR_COST = 1.0

FONTS_DIR_PATTERN = pathjoin(DATASET_DIR, 'fonts/*/*.ttf')

BEST_MODEL = pathjoin(MODELS_DIR, 'new-epoch102-hebnet__shape-32X32_f1-32_k1-3_f2-64_k2-2_f3-96_k3-2_d1-512-384_func-relu.model')  # noqa
BEST_MODEL_TYPE = 'hebnet'

HEBCHARS = u'אבגדהוזחטיכלמנסעפצקרשתןםץףך'
HEBCHARS_DFUS_KTIV_SIMILAR = u'בהוחיךכםןסקרת'
DIGITS = u'0123456789'

# mapping and reverse mapping of index to character
idx2char = HEBCHARS
char2idx = {k: i for i, k in enumerate(idx2char)}
N_CLASSES = len(idx2char)
