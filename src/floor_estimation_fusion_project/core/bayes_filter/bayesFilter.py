"""
Main script for data fusion by bayes filter
10/2/2021
Martin Havelka
"""

import numpy as np
import pylab as plt


class BayesFilter(object):
    """
    class that compute final probability by Bayes filter
    """

    def __init__(self):
        self.acc_diff_floor = 0
        self.camera_floor = 0

        self.floor_count = 9  # max of building floor number (1-9/ 0-8)
        self.bel = [1.0/self.floor_count] * self.floor_count
        print(self.bel)
        self.bel_new = []
        self.floor_probability_result = []

    def createProbDictionary(self):
        """
        dictionary of probabilities for camera and acc diff_floor
        """
        # use matrix for 9 floor buildings

        self.acc_probab_table = np.array([
            [0.93, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            [0.02, 0.81, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
            [0.02, 0.04, 0.71, 0.11, 0.02, 0.02, 0.02, 0.02, 0.02],
            [0.03, 0.03, 0.03, 0.75, 0.06, 0.03, 0.03, 0.03, 0.03],
            [0.03, 0.03, 0.03, 0.08, 0.68, 0.08, 0.03, 0.03, 0.03],
            [0.03, 0.03, 0.03, 0.03, 0.35, 0.19, 0.26, 0.03, 0.03],
            [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.33, 0.08],
            [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
            [0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11]
        ])

        self.camera_probab_table = np.array([
            [0.899, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004],
            [0.003, 0.867, 0.003, 0.003, 0.114, 0.003, 0.003, 0.003, 0.003],
            [0.005, 0.005, 0.957, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005],
            [0.009, 0.009, 0.009, 0.930, 0.009, 0.009, 0.009, 0.009, 0.009],
            [0.007, 0.007, 0.007, 0.007, 0.945, 0.007, 0.007, 0.007, 0.007],
            [0.003, 0.003, 0.003, 0.003, 0.006, 0.974, 0.003, 0.003, 0.003],
            [0.007, 0.020, 0.007, 0.027, 0.007, 0.020, 0.898, 0.007, 0.007],
            [0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.926, 0.009],
            [0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.926]
        ])

    def countProbability(self):
        """
        creates bayesian network for data fusion

        # general inputs
        self.acc_diff_floor - measurement from accelerometer
        self.camera_floor - measurement from camera
        """
        self.bel_new = []
        # probability table starts from 1, not from 0.
        # :TODO change probability table and add 0 row
        camera_floor = self.camera_floor - 1
        for i in range(self.floor_count):
            p = 0
            # ====== PREDICTION ======= #
            for j in range(self.floor_count):
                p += self.acc_probab_table[abs(self.acc_diff_floor), abs(j-i)] * self.bel[j]
            # ==== CORECTION ====== #
            self.bel_new.append(self.camera_probab_table[camera_floor, i]*p)

        # Bel calculation using Eta => normalization
        for i in range(len(self.bel)):
            self.bel_new[i] /= sum(self.bel_new)

        print(self.bel_new)

        # actualize bel() for next iteration
        self.bel = np.copy(self.bel_new[:])

        # save bel to the array
        self.floor_probability_result.append(self.bel_new)

    def spin(self, acc_diff_floor, camera_floor):
        """
        main function of BayesNet class
        - count probability of data fusion from acc and camera floor estimation

        input: acc_diff_floor - int value
        input: camera_floor - int value

        output: floor_pobability result - ndarray
        """
        self.acc_diff_floor = acc_diff_floor
        self.camera_floor = camera_floor

        self.createProbDictionary()
        self.countProbability()

        return self.floor_probability_result

    def plotProbability(self):
        """
        function plot probability belives for given floors during 5 steps of ride
        """
        plt.figure()
        for attempt in [0, 1, 2, 3, 4]:
            plt.plot(self.floor_probability_result[attempt])
        plt.title('Belive that robot is in given floor')
        plt.show()
