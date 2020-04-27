#################################
# Your name: Shany Shmueli
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import numpy.random
import sklearn.preprocessing
# from sklearn.datasets import fetch_mldata
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import random
# from tensorflow import keras



NUM_OF_LABELS = 10

"""
Assignment 3 question 2 skeleton.

Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper_hinge():
    mnist = fetch_openml('mnist_784')
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def helper_ce():
    mnist = fetch_openml('mnist_784')
    data = mnist['data']
    labels = mnist['target']

    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:8000] != 'a'))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[8000:10000] != 'a'))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = labels[train_idx[:6000]]

    validation_data_unscaled = data[train_idx[6000:8000], :].astype(float)
    validation_labels = labels[train_idx[6000:8000]]

    test_data_unscaled = data[8000 + test_idx, :].astype(float)
    test_labels = labels[8000 + test_idx]

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def SGD_hinge(data, labels, C, eta_0, T):
    w = np.zeros(np.size(data[0]))
    for epoch in range(T):
        i = np.random.randint(0, len(data))
        x = data[i]
        y = labels[i]
        # Perform the gradient descent update
        eta_t = eta_0 / (epoch + 1)
        if y * (np.dot(w, x)) < 1:
            w = (1 - eta_t) * w + eta_t * C * y * x
        else:
            w = (1 - eta_t) * w
    return w


def calc_p(w, x, y):
    return (np.exp(np.dot(w[int(y)], x)) / np.sum([np.exp(np.dot(w[i], x)) for i in range(NUM_OF_LABELS)])) * x


def SGC_ce_get_gradient(w, x, y):
    grad = []
    for i in range(NUM_OF_LABELS):
        if int(y) == i:  # weight of the correct label
            grad.append(calc_p(w, x, y) - x)
        else:
            grad.append(calc_p(w, x, y))
    return np.array(grad)


def SGD_ce(data, labels, eta_0, T):
    w = np.zeros((NUM_OF_LABELS, np.size(data[0])))
    for epoch in range(T):
        print('epoch {}'.format(epoch))
        i = np.random.randint(0, len(data))
        x = data[i]
        y = labels[i]
        # Perform the gradient descent update
        eta_t = eta_0 / (epoch + 1)
        grad = SGC_ce_get_gradient(w, x, y)
        # w = (1 - eta_t) * w - eta_t * grad
        # w = w - eta_t * grad
        w = w - eta_0 * grad
    return w


def predict_x_by_w(x, w):
    if np.dot(x, w) >= 0:
        return 1
    return -1


def calc_sgd_hinge_accuracy(data, labels, w):
    accuracy_num = 0
    for i in range(len(labels)):
        accuracy_num += int((predict_x_by_w(data[i], w) == labels[i]) == True)
    return accuracy_num / len(labels)


def predict_x_by_w_multiclass(x, w):
    return np.argmax(np.dot(x, w))


def calc_sgd_ce_accuracy(data, labels, w):
    accuracy_num = 0
    for i in range(len(labels)):
        accuracy_num += int((predict_x_by_w_multiclass(data[i], w) == labels[i]) == True)
    return accuracy_num / len(labels)


def create_eta_0_options():
    arr = np.array(np.arange(-5, 6), dtype='float32')
    return [10 ** x for x in arr]


def create_C_options():
    arr = np.array(np.arange(-10, 10), dtype='float32')
    return [10 ** x for x in arr]


def section_1a():
    T = 1000
    C = 1
    num_of_runs = 10
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_hinge()
    eta_0_options = create_eta_0_options()
    avg_accuracy_list = []
    for eta_0 in eta_0_options:
        accuracy_list = []
        for i in range(num_of_runs):
            w = SGD_hinge(train_data, train_labels, C, eta_0, T)
            accuracy_list.append(calc_sgd_hinge_accuracy(validation_data, validation_labels, w))
        avg_accuracy_list.append(np.average(accuracy_list))

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.grid()
    plt.plot(eta_0_options, avg_accuracy_list, 'o', color='#44BAEC', markersize=2)
    plt.xticks(eta_0_options)

    plt.xlabel('eta_0')
    plt.ylabel('Accuracy')
    plt.ylim(np.min(avg_accuracy_list) - (1 - np.max(avg_accuracy_list)), 1)
    plt.xlim(eta_0_options[0], eta_0_options[len(eta_0_options)-1])
    plt.savefig('results/section_1a.png')
    print('Best eta_0: {}'.format(eta_0_options[np.argmax(avg_accuracy_list)]))


def section_1b():
    best_eta_0 = 1
    T = 1000
    num_of_runs = 10
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_hinge()
    C_options = create_C_options()
    avg_accuracy_list = []
    for C in C_options:
        accuracy_list = []
        for i in range(num_of_runs):
            w = SGD_hinge(train_data, train_labels, C, best_eta_0, T)
            accuracy_list.append(calc_sgd_hinge_accuracy(validation_data, validation_labels, w))
        avg_accuracy_list.append(np.average(accuracy_list))

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.grid()
    plt.plot(C_options, avg_accuracy_list, 'o', color='#44BAEC', markersize=2)
    plt.xticks(C_options)

    plt.xlabel('C values')
    plt.ylabel('Accuracy')
    plt.ylim(np.min(avg_accuracy_list) - (1 - np.max(avg_accuracy_list)), 1)
    plt.xlim(C_options[0], C_options[len(C_options)-1])
    plt.savefig('results/section_1b.png')
    print('Best C: {}'.format(C_options[np.argmax(avg_accuracy_list)]))


def section_1c():
    best_eta_0 = 1
    best_C = 0.0001
    T = 20000
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_hinge()
    w = SGD_hinge(train_data, train_labels, best_C, best_eta_0, T)
    accuracy = calc_sgd_hinge_accuracy(test_data, test_labels, w)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest', cmap='plasma')
    plt.axis('off')
    plt.savefig('results/section_1c.png')
    print('Accuracy: {}'.format(accuracy))


def section_2a():
    T = 1000
    num_of_runs = 10
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_ce()
    eta_0_options = create_eta_0_options()
    avg_accuracy_list = []
    for eta_0 in eta_0_options:
        accuracy_list = []
        for i in range(num_of_runs):
            w = SGD_ce(train_data, train_labels, eta_0, T)
            accuracy_list.append(calc_sgd_ce_accuracy(validation_data, validation_labels, w))
        avg_accuracy_list.append(np.average(accuracy_list))

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.grid()
    plt.plot(eta_0_options, avg_accuracy_list, 'o', color='#44BAEC', markersize=2)
    plt.xticks(eta_0_options)

    plt.xlabel('eta_0')
    plt.ylabel('Accuracy')
    plt.ylim(np.min(avg_accuracy_list) - (1 - np.max(avg_accuracy_list)), 1)
    plt.xlim(eta_0_options[0], eta_0_options[len(eta_0_options)-1])
    plt.savefig('results/section_2a.png')
    print('Best eta_0: {}'.format(eta_0_options[np.argmax(avg_accuracy_list)]))


if __name__ == "__main__":
    section_2a()

