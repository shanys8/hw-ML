#################################
# Your name: Shany Shmueli
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import matplotlib.pyplot as plt
import numpy.random
import random
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing
from columnar import columnar


"""
Assignment 3 question 1 skeleton.

Please use the provided function signature for the perceptron implementation.
Feel free to add functions and other code, and submit this file with the name perceptron.py
"""


def helper():
    # mnist = fetch_mldata('MNIST original', transpose_data=True, data_home='files')
    mnist = fetch_mldata('MNIST original', data_home='files')
    data = mnist['data']
    labels = mnist['target']
    # neg, pos = "0", "8"
    neg, pos = 0, 8
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])
    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos)*2-1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos)*2-1

    test_data_unscaled = data[60000+test_idx, :].astype(float)
    test_labels = (labels[60000+test_idx] == pos)*2-1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def perceptron(samples, labels):
    """
    returns: nd array of shape (data.shape[1],) or (data.shape[1],1) representing the perceptron classifier
    """
    num_of_samples = len(labels)
    samples = normalize_data_to_unit_length(samples)
    w = np.zeros(np.size(samples[0]))  # we assume samples is not empty
    for t in range(num_of_samples):
        x_t = samples[t]
        y_t = labels[t]
        y_t_predicted = predict_x_by_w(x_t, w)
        if y_t_predicted != y_t:
            w += y_t * x_t

    return w


#################################

# Place for additional code

#################################

def normalize_data_to_unit_length(data):
    return sklearn.preprocessing.normalize(data, norm='l2')


def print_norms(samples, labels):
    for i in range(len(labels)):
        print('{} => {}'.format(np.linalg.norm(samples[i]), labels[i]))


def random_order(samples, labels):
    z = list(zip(samples, labels))
    random.shuffle(z)
    samples, labels = zip(*z)
    return samples, labels


def predict_x_by_w(x, w):
    if np.dot(x, w) >= 0:
        return 1
    return -1
    # return np.sign(np.dot(x, w))


def calc_accuracy(w, test_data, test_labels):
    accuracy_num = 0
    for i in range(len(test_labels)):
        accuracy_num += int((predict_x_by_w(test_data[i], w) == test_labels[i]) == True)
    return accuracy_num / len(test_labels)


def get_accuracy_results(accuracy_array):
    return {'mean': np.mean(accuracy_array),
            '5_percentile': np.percentile(accuracy_array, 5),
            '95_percentile': np.percentile(accuracy_array, 95)}


def print_table(samples_num, mean_list, five_percentile_list, ninety_five_percentile):
    headers = ['Num of Samples', 'Mean', '5% Percentile', '95% Percentile']
    n = len(samples_num)

    data = []
    for i in range(n):
        data.append([samples_num[i], mean_list[i], five_percentile_list[i], ninety_five_percentile[i]])

    table = columnar(data, headers)
    print(table)


def section_a():
    train_data, train_labels, _, _, test_data, test_labels = helper()
    samples_num = [5, 10, 50, 100, 500, 1000, 5000]
    T = 100
    accuracy_results = []
    for n in samples_num:
        accuracy_array = []
        first_n_samples = train_data[:n]
        first_n_labels = train_labels[:n]
        for i in range(T):
            samples, labels = random_order(first_n_samples, first_n_labels)
            w = perceptron(samples, labels)
            accuracy = calc_accuracy(w, test_data, test_labels)
            accuracy_array.append(accuracy)
        accuracy_results.append(get_accuracy_results(accuracy_array))

    mean_list = [res['mean'] for res in accuracy_results]
    five_percentile_list = [res['5_percentile'] for res in accuracy_results]
    ninety_five_percentile = [res['95_percentile'] for res in accuracy_results]

    print_table(samples_num, mean_list, five_percentile_list, ninety_five_percentile)


def section_b():
    train_data, train_labels, _, _, _, _ = helper()
    w = perceptron(train_data, train_labels)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest', cmap='plasma')
    plt.axis('off')
    plt.savefig('results/section_b.png')
    return


def section_c():
    train_data, train_labels, _, _, test_data, test_labels = helper()
    w = perceptron(train_data, train_labels)
    accuracy = calc_accuracy(w, test_data, test_labels)
    print('Accuracy: {}'.format(accuracy))


def section_d():
    train_data, train_labels, _, _, test_data, test_labels = helper()
    w = perceptron(train_data, train_labels)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest', cmap='plasma')
    plt.axis('off')
    plt.savefig('results/section_d_classifier')
    count = 1
    for i in range(len(test_labels)):
        if count > 10:
            break
        if predict_x_by_w(test_data[i], w) != test_labels[i]:
            print('image {} should classified as {} but was classified as {}'.format(i, test_labels[i], predict_x_by_w(test_data[i], w)))
            plt.imshow(np.reshape(test_data[i], (28, 28)), interpolation='nearest', cmap='plasma')
            plt.axis('off')
            plt.savefig('results/section_d_bad_classification_{}.png'.format(i))
            count += 1


def print_several_test_images():
    train_data, train_labels, _, _, test_data, test_labels = helper()
    for i in range(10):
        plt.imshow(np.reshape(test_data[i], (28, 28)), interpolation='nearest', cmap='plasma')
        plt.axis('off')
        if test_labels[i] == 1:
            plt.savefig('results/eight_figures_{}.png'.format(i))
        else:
            plt.savefig('results/zero_figures_{}.png'.format(i))


if __name__ == "__main__":
    section_c()
