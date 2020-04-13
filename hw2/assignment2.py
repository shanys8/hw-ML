#################################
# Your name:
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals
import math


def label_1_with_probability(probability):
    random = np.random.uniform(0, 1)
    if random < probability:
        return 1
    else:
        return 0


def determine_label(x):
    if 0 <= x <= 0.2 or 0.4 <= x <= 0.6 or 0.8 <= x <= 1:
        return label_1_with_probability(0.8)
    return label_1_with_probability(0.1)


def determine_label_by_h(x, h):
    for i in range(len(h)):
        min_interval, max_interval = h[i]
        if min_interval <= x <= max_interval:
            return 1
    return 0


def calc_penalty(m, k):
    delta = 0.1
    d = 2 * k  # as calculated in question 3 as the VCdim of h
    return math.sqrt((8 / m) * (np.log(4 / delta) + d * np.log((2 * m * math.e) / d)))


def intersection(interval1, interval2):
    if interval1[0] > interval2[0]:
        interval1, interval2 = interval2, interval1
    if interval1[1] <= interval2[0]:
        return 0
    return min(interval1[1], interval2[1]) - interval2[0]


def union(h):
    result = 0
    for interval in h:
        result += interval[1] - interval[0]
    return result


def h_zero_intervals(h):
    h.insert(0, (0, 0))
    h.append((1, 1))
    zero_intervals = []
    for i in range(len(h) - 1):
        zero_intervals.append((h[i][1], h[i + 1][0]))
    return zero_intervals


def calc_true_error(h):
    # intervals with labels 1
    first_1_interval = [0, 0.2]
    second_1_interval = [0.4, 0.6]
    third_1_interval = [0.8, 1]
    # intervals with labels 0
    first_0_interval = [0.2, 0.4]
    second_0_interval = [0.6, 0.8]
    # sum of the lengths of intervals for which h labels 1 and intersect with ([0, 0.2] union [0.4, 0.6] union [0.8, 1])
    intersection_with_1_intervals = 0
    # sum of the lengths of intervals for which h labels 0 and intersect with ([0.2, 0.4] union [0.6, 0.8])
    intersection_with_0_intervals = 0

    intervals_of_h_with_label_1 = h
    intervals_of_h_with_label_0 = h_zero_intervals(h)

    union_of_h_with_label_1 = union(intervals_of_h_with_label_1)
    union_of_h_with_label_0 = union(intervals_of_h_with_label_0)

    for interval in intervals_of_h_with_label_1:
        intersection_with_1_intervals += intersection(interval, first_1_interval)
        intersection_with_1_intervals += intersection(interval, second_1_interval)
        intersection_with_1_intervals += intersection(interval, third_1_interval)

    for interval in intervals_of_h_with_label_0:
        intersection_with_0_intervals += intersection(interval, first_0_interval)
        intersection_with_0_intervals += intersection(interval, second_0_interval)

    res = 0.2 * intersection_with_1_intervals + \
          0.9 * (union_of_h_with_label_1 - intersection_with_1_intervals) + \
          0.1 * intersection_with_0_intervals + \
          0.8 * (union_of_h_with_label_0 - intersection_with_0_intervals)

    return res


def calc_empirical_error(samples, h):
    y_samples = samples[:, 1]
    x_samples = samples[:, 0]
    h_labels = [determine_label_by_h(x, h) for x in x_samples]
    error_num = np.sum(y_samples != h_labels)
    return error_num / np.size(y_samples)


def x_in_some_interval(x, h):
    for interval in h:
        if (x >= interval[0]) and (x <= interval[1]):
            return True
    return False


def calc_holdout_error(m, h, holdout_samples):
    error_num = 0
    for i in range(len(holdout_samples)):
        if x_in_some_interval(holdout_samples[i][0], h):
            if holdout_samples[i][1] == 0:  # case h(x)=1 but holdout label is 0
                error_num += 1
        else:
            if holdout_samples[i][1] == 1:  # case h(x)=0 but holdout label is 1
                error_num += 1
    return error_num / m


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        x_samples = np.random.uniform(0, 1, m)
        x_samples = np.sort(x_samples)
        y_samples = np.array(list(map(determine_label, x_samples)))
        return np.vstack((x_samples, y_samples)).T

    def draw_sample_intervals(self, m, k):
        """
        Plots the data as asked in (a) i ii and iii.
        Input: m - an integer, the size of the data sample.
               k - an integer, the maximum number of intervals.

        Returns: None.
        """
        samples = self.sample_from_D(m)
        best_intervals, besterror = intervals.find_best_interval(samples[:, 0], samples[:, 1], k)
        plt.plot(samples[:, 0], samples[:, 1], 'o', color='#44BAEC', markersize=2)
        for i in range(k):
            plt.plot(best_intervals[i], [1.05, 1.05], color='#A437F7') # plot intervals in middle of graph
        plt.ylim(-0.1, 1.1)
        plt.xlim(0, 1)
        plt.savefig('results/section_a.png')
        return

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        m_list = np.arange(m_first, m_last + step, step)
        avg_empirical_error_list = []
        avg_true_error_list = []
        for m in m_list:
            print(m)
            empirical_error_list = []
            true_error_list = []
            for i in range(T):
                samples = self.sample_from_D(m)
                h, best_error = intervals.find_best_interval(samples[:, 0], samples[:, 1], k)
                empirical_error = calc_empirical_error(samples, h)
                # empirical_error == best_error/m
                true_error = calc_true_error(h)
                empirical_error_list.append(empirical_error)
                true_error_list.append(true_error)
            avg_empirical_error_list.append(np.average(empirical_error_list))
            avg_true_error_list.append(np.average(true_error_list))

        plt.plot(m_list, avg_empirical_error_list, label="avg empirical errors")
        plt.plot(m_list, avg_true_error_list, label="avg true errors")
        plt.legend()
        plt.savefig('results/section_c.png')
        return np.vstack((avg_empirical_error_list, avg_true_error_list)).T

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        k_list = np.arange(k_first, k_last + step, step)
        samples = self.sample_from_D(m)
        empirical_error_list = []
        true_error_list = []
        for k in k_list:
            print(k)
            h, _ = intervals.find_best_interval(samples[:, 0], samples[:, 1], k)
            empirical_error = calc_empirical_error(samples, h)
            true_error = calc_true_error(h)
            empirical_error_list.append(empirical_error)
            true_error_list.append(true_error)

        plt.plot(k_list, empirical_error_list, label="empirical errors")
        plt.plot(k_list, true_error_list, label="true errors")
        plt.legend()
        plt.savefig('results/section_d.png')
        best_k = k_list[np.argmin(empirical_error_list)]
        print('best k {}'.format(best_k))
        return best_k

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Runs the experiment in (d).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        k_list = np.arange(k_first, k_last + step, step)
        samples = self.sample_from_D(m)
        empirical_error_list = []
        true_error_list = []
        penalty_list = []
        sum_empirical_error_and_penalty_list = []
        for k in k_list:
            print(k)
            h, _ = intervals.find_best_interval(samples[:, 0], samples[:, 1], k)
            empirical_error = calc_empirical_error(samples, h)
            true_error = calc_true_error(h)
            empirical_error_list.append(empirical_error)
            true_error_list.append(true_error)
            penalty = calc_penalty(m, k)
            penalty_list.append(penalty)
            sum_empirical_error_and_penalty_list.append(empirical_error+penalty)

        plt.plot(k_list, empirical_error_list, label="empirical errors")
        plt.plot(k_list, true_error_list, label="true errors")
        plt.plot(k_list, penalty_list, label="penalties")
        plt.plot(k_list, sum_empirical_error_and_penalty_list, label="penalty + empirical error")
        plt.legend()
        plt.savefig('results/section_e.png')
        best_k = k_list[np.argmin(sum_empirical_error_and_penalty_list)]
        print('best_k {}'.format(best_k))
        return best_k

    def cross_validation(self, m, T):
        """Finds a k that gives a good test error.
        Chooses the best hypothesis based on 3 experiments.
        Input: m - an integer, the size of the data sample.
               T - an integer, the number of times the experiment is performed.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """

        samples = self.sample_from_D(m)
        best_k_list = []
        for i in range(T):
            print('{} out of T={}'.format(i, T))
            np.random.shuffle(samples)
            holdout_samples = samples[:m // 5, :]
            train_samples = samples[m // 5:, :]
            train_samples = np.asarray(sorted(train_samples, key=lambda a_entry: a_entry[0]))
            min_k = 1
            min_k_holdout_error = 1
            for k in range(1, 10):
                print('k={}'.format(k))
                h, _ = intervals.find_best_interval(train_samples[:, 0], train_samples[:, 1], k)
                holdout_error = calc_holdout_error(m // 5, h, holdout_samples)
                if holdout_error < min_k_holdout_error:
                    min_k_holdout_error = holdout_error
                    min_k = k
            print('min_k {}'.format(min_k))
            best_k_list.append(min_k)

        print(best_k_list)
        counts = np.bincount(best_k_list)
        return np.argmax(counts)



if __name__ == '__main__':
    ass = Assignment2()

    # ass.draw_sample_intervals(100, 3)
    # ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    # ass.experiment_k_range_erm(1500, 1, 10, 1)
    # ass.experiment_k_range_srm(1500, 1, 10, 1)
    # ass.cross_validation(1500, 3)

