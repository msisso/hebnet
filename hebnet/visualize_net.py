# coding=utf-8
# Imports
import matplotlib.pyplot as plt
from matplotlib import cm, gridspec

from keras.layers.core import Activation
import theano
from hebnet import dnn_nets
from scipy.misc import imread
from hebnet.utils import dataset

fname = './models/epoch%d-hebnet_new-dataset_shape-32X32_f1-32_k1-3_f2-64_k2-2_f3-96_k3-2_d1-512-384_func-relu.model'  # noqa
img_fname = '../dataset/human/train/א-set7-new_sets-544.png'


def get_model(weights_fname, epoch):
    # build the net
    model = dnn_nets.build_hebnet(nb_classes=38, img_shape=(1, 32, 32))
    if epoch > 0:
        model.load_weights(weights_fname % epoch)

    return model


def get_fig_size(num_filters):
    # get the image dimensions for different number of conv filters
    if num_filters == 32:
        return 4, 8,  # 4X8 images == 32 filters

    if num_filters == 64:
        return 8, 8,  # 8X8 images == 64 filters

    if num_filters == 96:
        return 12, 8,  # 12X8 images == 96 filters

    raise ValueError('bad num_filters=%s' % num_filters)


def get_fig_dim(num_filters):
    if num_filters == 32:
        return 16, 8,  # 4X8 images == 32 filters

    if num_filters == 64:
        return 16, 16,  # 8X8 images == 64 filters

    if num_filters == 96:
        return 16, 24,  # 8X8 images == 96 filters

    raise ValueError('bad num_filters=%s for dim' % num_filters)


def plot_layer(convolutions):
    # show the 32 convolutions of the image @ conv
    dim = get_fig_dim(convolutions.shape[1])
    fig = plt.figure(figsize=dim)

    # disable spacing between images
    gs1 = gridspec.GridSpec(*dim)  # 16,8 for 32
    gs1.update(wspace=0.000025, hspace=0.00005)

    for i, convolution in enumerate(convolutions[0]):
        fig_size = get_fig_size(convolutions.shape[1])
        fig_size = fig_size + (i + 1, )
        a = fig.add_subplot(*fig_size)
        imgplot = plt.imshow(convolution, cmap=cm.Greys_r)
        imgplot.axes.axis('off')
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.set_aspect('equal')

    fig.tight_layout()
    return fig


def conv_img_with_layer(model, img, layer_num):
    # get all activation layers
    convout_layers = [layer for layer in model.layers
                      if isinstance(layer, Activation)]

    print('got %d activation layers' % len(convout_layers))

    # get only the output up to the requested layer
    layer_output = convout_layers[layer_num].get_output(train=False)
    convout1_f = theano.function([model.get_input(train=False)], layer_output)

    # get the image @ convolved with the 32 filters
    convolutions = convout1_f(img.reshape(1, 1, 32, 32))

    return convolutions


def visualize(epoch, layer):
    # load the model and weights
    model = get_model(fname, epoch)

    # load some image to visualize
    img = dataset.normalize_image(imread(img_fname, flatten=True))

    # hook the conv layer
    convolutions = conv_img_with_layer(model, img, layer)

    fig = plot_layer(convolutions)
    fig.savefig('../net-images/net_vis_layer-%d_epoch-%d.png' % (layer, epoch),
                bbox_inches='tight')
    del model


if __name__ == '__main__':
    for l in range(4):
        for e in [12]:
            print('layer%d epoch%d' % (l, e))
            visualize(e, l)
