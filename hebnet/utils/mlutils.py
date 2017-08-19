from __future__ import unicode_literals

import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model as KerasModel
from keras.utils import np_utils
from sklearn import metrics


def save_conf_mat(conf_arr, stats, labels=None, fname=None):
    labels = labels or []
    fname = fname or 'conf_matrix.png'
    if not fname.strip().lower().endswith('.png'):
        fname = '%s.png' % fname

    norm_conf = []
    for i in conf_arr:
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            try:
                tmp_arr.append(float(j)/float(a))
            except ZeroDivisionError:
                tmp_arr.append(0.0)
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    plt.rc('font', **{'family': 'Arial', 'weight': 'bold', 'size': 8})
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                    interpolation='nearest')

    width = len(conf_arr)
    height = len(conf_arr[0])
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    fig.colorbar(res)

    stats_txt = '\n'.join('%s: %.7f' % s for s in sorted(stats.iteritems()))
    plt.text(0.05, -1.0, stats_txt)
    print('saving stats image to: %s' % fname)
    plt.savefig(fname, format='png')


def predict_class(clf, X_test, Y_test, labels=None, stats_fname=None):
    """using the `clf` predict the test set, print results.
    returns the prediction.
    """
    expected = Y_test
    if isinstance(clf, KerasModel):
        char_probs = clf.predict(X_test)
        predicted = np.argmax(char_probs, axis=1)

        if len(Y_test.shape) > 1:
            expected = np.argmax(Y_test, axis=1)
    else:
        predicted = clf.predict(X_test)

    conf_mat = metrics.confusion_matrix(
        expected, predicted, labels=range(len(labels))
    )

    stats = {
        'Accuracy': metrics.accuracy_score(expected, predicted),
        'F1': metrics.f1_score(expected, predicted, average='weighted'),
        'Precision': metrics.precision_score(expected, predicted,
                                             average='weighted'),
        'Recall': metrics.recall_score(expected, predicted,
                                       average='weighted')
    }
    print('Accuracy: %f' % stats['Accuracy'])
    print('F1: %f' % stats['F1'])
    print('percision: %f' % stats['Precision'])
    print('recall: %f' % stats['Recall'])

    save_conf_mat(conf_mat, stats, labels, stats_fname)

    return predicted


def num2categorical(y_train, y_valid, y_test, nb_classes):
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_valid = np_utils.to_categorical(y_valid, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return Y_train, Y_valid, Y_test
