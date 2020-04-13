import numpy as np
from sklearn.datasets import fetch_mldata
from matplotlib import pyplot as plt
from collections import Counter

mnist = fetch_mldata('MNIST original', transpose_data=True, data_home='files')
data = mnist['data']
labels = mnist['target']

idx = np.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]


# section_a
def knn(train, train_labels, query_image, k):
    distance_label_tuple = [(0, 0) for _ in range(len(train))]
    for i in range(len(distance_label_tuple)):
        distance_label_tuple[i] = (np.linalg.norm(train[i] - query_image), train_labels[i])
    distance_label_tuple.sort(key=lambda x: x[0])
    k_nn = distance_label_tuple[:k]
    nearest_labels = Counter([x[1] for x in k_nn])
    value, _ = nearest_labels.most_common()[0]
    return value


def section_d():
    k = 1
    accuracy_percentages_list = []

    samples_num_list = np.arange(100, 5100, 100)
    for samples_num in samples_num_list:
        partial_train_images = train[:samples_num, :]
        partial_train_labels = train_labels[:samples_num]
        loss_list = []
        for i in range(len(test_labels)):
            knn_value = knn(partial_train_images, partial_train_labels, test[i], k)
            loss_list.append(0 if knn_value == test_labels[i] else 1)

        accuracy_percentage = round(((len(test_labels) - np.sum(loss_list)) / len(test_labels)) * 100, 2)
        accuracy_percentages_list.append(accuracy_percentage)
        print('Accuracy Percentage for samples_num = {} is: {}'.format(samples_num, accuracy_percentage))

    plt.plot(samples_num_list, accuracy_percentages_list)
    plt.show()


def section_b():
    k = 10
    samples_num = 1000
    first_1000_train_images = train[:samples_num, :]
    first_1000_train_labels = train_labels[:samples_num]
    loss_list = []

    for i in range(len(test_labels)):
        knn_value = knn(first_1000_train_images, first_1000_train_labels, test[i], k)
        loss_list.append(0 if knn_value == test_labels[i] else 1)

    accuracy_percentage = round(((samples_num - np.sum(loss_list)) / samples_num) * 100, 2)
    print('Accuracy Percentage for k = 10 is: {}'.format(accuracy_percentage))


def section_c():
    samples_num = 1000
    first_1000_train_images = train[:samples_num, :]
    first_1000_train_labels = train_labels[:samples_num]
    loss_list = [0 for _ in range(samples_num)]
    accuracy_percentages_list = []

    k_max = 100
    for k in range(k_max):
        for i in range(samples_num):
            knn_value = knn(first_1000_train_images, first_1000_train_labels, test[i], k+1)
            loss_list[i] = 0 if knn_value == test_labels[i] else 1

        accuracy_percentage = round(((samples_num - np.sum(loss_list)) / samples_num) * 100, 2)
        accuracy_percentages_list.append(accuracy_percentage)
        print('Accuracy Percentage for k = {} is: {}'.format(k+1, accuracy_percentage))

    plt.plot(list(range(1, k_max + 1)), accuracy_percentages_list)
    plt.show()


def main():
    section_d()
    return


if __name__ == "__main__":
    main()
