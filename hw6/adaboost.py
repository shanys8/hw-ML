#################################
# Your name:
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
from matplotlib import pyplot as plt
import numpy as np
import math
import sys
from process_data import parse_data
from tqdm import tqdm
import time

np.random.seed(7)


def classify_by_h(h, S):
    h_index, h_theta, h_pred = h
    preds = np.zeros(len(S))
    for i, s in enumerate(S):
        if s[h_index] <= h_theta:
            preds[i] = h_pred
        else:
            preds[i] = 0 - h_pred
    return preds


def run_adaboost(X_train, y_train, T):
    """
    Returns: 

        hypotheses : 
            A list of T tuples describing the hypotheses chosen by the algorithm. 
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is 
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals : 
            A list of T float values, which are the alpha values obtained in every 
            iteration of the algorithm.
    """

    hypotheses_list = []
    alpha_vals = []

    D = np.array([1/len(X_train)]*len(X_train))

    for t in tqdm(range(T)):
        print('{} out of {}'.format(t, T))
        curr_h = select_hypotheses(D, X_train, y_train)
        hypotheses_list.append(curr_h)
        preds = classify_by_h(curr_h, X_train)
        error = np.sum(D[preds != y_train])
        alpha = calc_alpha(error)
        alpha_vals.append(alpha)
        arr = D * (np.e ** (-alpha * y_train * preds))
        D = arr / np.sum(arr)

    return hypotheses_list, alpha_vals


##############################################
# You can add more methods here, if needed.


def select_hypotheses(D, X_train, y_train):
    F_star_p = np.inf
    theta_star_p = 0
    j_star_p = 0
    F_star_m = np.inf
    theta_star_m = 0
    j_star_m = 0
    for j in range(X_train.shape[1]):
        # print('{} out of {}'.format(j, X_train.shape[1]))
        sorted_x = np.sort(X_train[:, j])
        sorted_ind = X_train[:, j].argsort()
        sorted_x = np.append(sorted_x, sorted_x[-1] + 1)
        sorted_y = y_train[sorted_ind]
        sorted_D = D[sorted_ind]
        F_positive = np.sum(D[y_train == 1])
        F_negative = np.sum(D[y_train == -1])
        if F_positive < F_star_p:
            F_star_p = F_positive
            theta_star_p = sorted_x[0] - 1
            j_star_p = j
        if F_negative < F_star_m:
            F_star_m = F_negative
            theta_star_m = sorted_x[0] - 1
            j_star_m = j
        for i in range(len(sorted_x) - 1):
            F_positive = F_positive - sorted_y[i]*sorted_D[i]
            F_negative = F_negative + sorted_y[i] * sorted_D[i]
            if sorted_x[i] != sorted_x[i + 1]:
                if F_positive < F_star_p:
                    F_star_p = F_positive
                    theta_star_p = 0.5 * (sorted_x[i] + sorted_x[i + 1])
                    j_star_p = j
                if F_negative < F_star_m:
                    F_star_m = F_negative
                    theta_star_m = 0.5 * (sorted_x[i] + sorted_x[i + 1])
                    j_star_m = j
    if F_star_p < F_star_m:
        return j_star_p, theta_star_p, 1.0
    else:
        return j_star_m, theta_star_m, -1.0


def calc_alpha(error):
    return 0.5 * math.log((1 - error) / error)


def calc_loss(hypotheses, alpha_vals, X, y):
    predictions = np.zeros((len(hypotheses), len(X)))
    for i in range(len(hypotheses)):
        predictions[i, :] = alpha_vals[i] * classify_by_h(hypotheses[i], X)
    predictions = predictions.cumsum(axis=0)
    ex = np.e ** (-y * predictions)
    return ex.mean(axis=1)


def calc_error(hypotheses, alpha_vals, X, y):
    predictions = np.zeros((len(hypotheses), len(X)))
    for i in range(len(hypotheses)):
        predictions[i, :] = alpha_vals[i] * classify_by_h(hypotheses[i], X)
    predictions = predictions.cumsum(axis=0)
    predictions = np.sign(predictions)
    predictions[predictions == 0] = 1.
    return np.sum(predictions != y, axis=1) / len(y)


def section_a():
    start_time = time.time()

    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data
    T = 80
    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)
    training_err = calc_error(hypotheses, alpha_vals,X_train, y_train)
    test_err = calc_error(hypotheses, alpha_vals, X_test, y_test)
    plt.clf()
    x = np.arange(T)+1
    plt.plot(x, training_err, label='Train Error')
    plt.plot(x, test_err, label='Test Error')
    plt.legend()
    plt.xlabel("T Values")
    plt.ylabel("Error Rate")
    plt.savefig('results/section_1a.png')

    print("My program took", time.time() - start_time, "seconds to run")


def section_b():
    start_time = time.time()
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data
    T = 10
    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)
    words = {1: [], -1: []}
    i = 1
    for h in hypotheses:
        if h[2] == 1:
            print('hypothese {} is positive'.format(i))
            words[1].append(vocab[h[0]])
        else:
            print('hypothese {} is negative'.format(i))
            words[-1].append(vocab[h[0]])
        i += 1
    print(words)
    training_err = calc_error(hypotheses, alpha_vals, X_train, y_train)
    print(training_err)
    print("My program took", time.time() - start_time, "seconds to run")
# {1: ['bad', 'worst', 'boring', 'supposed', 'nothing'], -1: ['life', 'hilarious', 'performances', 'great', 'well']}


def section_c():
    start_time = time.time()

    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data
    T = 80
    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)
    training_exp_loss = calc_loss(hypotheses, alpha_vals, X_train, y_train)
    test_exp_loss = calc_loss(hypotheses, alpha_vals, X_test, y_test)
    plt.clf()
    x = np.arange(T)+1
    plt.plot(x, training_exp_loss, label='Train Exponential Loss')
    plt.plot(x, test_exp_loss, label='Test Exponential Loss')
    plt.legend()
    plt.xlabel("T Values")
    plt.ylabel("Exponential loss")
    plt.savefig('results/section_1c.png')

    print("My program took", time.time() - start_time, "seconds to run")

    ##############################################


def main():
    section_b()
    ##############################################
    # You can add more methods here, if needed.

    # plot_error(hypotheses, alpha_vals, T, X_train, y_train)
    # plot_error(hypotheses, alpha_vals, T, X_test, y_test)


    ##############################################

if __name__ == '__main__':
    main()



