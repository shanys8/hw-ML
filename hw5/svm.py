#################################
# Your name:
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

"""
Q4.1 skeleton.

Please use the provided functions signature for the SVM implementation.
Feel free to add functions and other code, and submit this file with the name svm.py
"""


# generate points in 2D
# return training_data, training_labels, validation_data, validation_labels
def get_points():
    X, y = make_blobs(n_samples=120, centers=2, random_state=0, cluster_std=0.88)
    return X[:80], y[:80], X[80:], y[80:]


def create_plot(X, y, clf):
    plt.clf()

    # plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.PiYG)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0] - 2, xlim[1] + 2, 30)
    yy = np.linspace(ylim[0] - 2, ylim[1] + 2, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])


def train_three_kernels(X_train, y_train, X_val, y_val):
    """
    Returns: np.ndarray of shape (3,2) :
                A two dimensional array of size 3 that contains the number of support vectors for each class(2) in the three kernels.
    """
    clf_rbf = svm.SVC(gamma='auto', C=1000, kernel='rbf')
    clf_linear = svm.SVC(gamma='auto', C=1000, kernel='linear')
    clf_quadratic = svm.SVC(gamma='auto', C=1000, kernel='poly')

    clf_rbf.fit(X_train, y_train)
    clf_linear.fit(X_train, y_train)
    clf_quadratic.fit(X_train, y_train)

    create_plot(X_val, y_val, clf_rbf)
    print('Num of SV for RBF: {}'.format(clf_rbf.n_support_))
    plt.savefig('results/section_1a_clf_rbf.png')

    create_plot(X_val, y_val, clf_linear)
    print('Num of SV for Linear: {}'.format(clf_linear.n_support_))
    plt.savefig('results/section_1a_clf_linear.png')

    create_plot(X_val, y_val, clf_quadratic)
    print('Num of SV for Quadratic: {}'.format(clf_quadratic.n_support_))
    plt.savefig('results/section_1a_clf_quadratic.png')

    return np.concatenate((clf_rbf.n_support_[np.newaxis, :], clf_linear.n_support_[np.newaxis, :],
                             clf_quadratic.n_support_[np.newaxis, :]), axis=0)


def linear_accuracy_per_C(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    C_options = create_C_options()
    train_accuracy_list = []
    val_accuracy_list = []
    for C in C_options:
        clf_linear = svm.SVC(gamma='auto', C=C, kernel='linear')
        clf_linear.fit(X_train, y_train)
        train_accuracy_list.append(clf_linear.score(X_train, y_train))
        val_accuracy_list.append(clf_linear.score(X_val, y_val))

    print('Best C: {} with accuracy on the validation data: {}'.format(C_options[np.argmax(val_accuracy_list)],
                                                                       val_accuracy_list[np.argmax(val_accuracy_list)]))
    print('train_accuracy_list')
    print(train_accuracy_list)
    print('val_accuracy_list')
    print(val_accuracy_list)

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.grid()
    plt.plot(C_options, train_accuracy_list, 'o', color='#44BAEC', markersize=4)
    plt.plot(C_options, val_accuracy_list, 'o', color='#e534eb', markersize=4)
    plt.xticks(C_options)

    plt.xlabel('C values')
    plt.ylabel('Accuracies')
    plt.legend() # TODO train acc and val acc
    plt.savefig('results/section_1b.png')


    return val_accuracy_list

def rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    # TODO: add your code here


def section_1a():
    training_data, training_labels, validation_data, validation_labels = get_points()
    sv_numbers = train_three_kernels(training_data, training_labels, validation_data, validation_labels)
    print(sv_numbers)
    return


def section_1b():
    training_data, training_labels, validation_data, validation_labels = get_points()
    val_accuracy_list = linear_accuracy_per_C(training_data, training_labels, validation_data, validation_labels)
    return


def create_C_options():
    arr = np.array(np.arange(-5, 5), dtype='float32')
    return [10 ** x for x in arr]


# def calc_accuracy(clf, X, y):
#     res = clf.score(X, y)
#     return res


if __name__ == "__main__":
    section_1b()
