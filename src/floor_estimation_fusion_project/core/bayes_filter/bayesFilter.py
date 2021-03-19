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

        self.floor_count = 15  # max of building floor number (-2 - 12)
        self.bel = [1.0/self.floor_count] * self.floor_count
        print(self.bel)
        self.bel_new = []
        self.floor_probability_result = []

    def createProbDictionary(self):
        """
        dictionary of probabilities for camera and acc diff_floor
        """
        # from floor diff = 1 to floor_diff = 13
        self.acc_probab_table = np.array([
            [0.951, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004],
            [0.011, 0.871, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011],
            [0.006, 0.006, 0.847, 0.083, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006],
            [0.004, 0.004, 0.039, 0.790, 0.127, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004],
            [0.005, 0.005, 0.005, 0.045, 0.662, 0.244, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005],
            [0.006, 0.006, 0.006, 0.006, 0.052, 0.676, 0.214, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006],
            [0.009, 0.009, 0.009, 0.009, 0.009, 0.046, 0.596, 0.266, 0.009, 0.009, 0.009, 0.009, 0.009],
            [0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.669, 0.228, 0.034, 0.007, 0.007, 0.007],
            [0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.129, 0.406, 0.366, 0.010, 0.010, 0.010],
            [0.022, 0.022, 0.022, 0.022, 0.022, 0.022, 0.022, 0.022, 0.111, 0.467, 0.200, 0.022, 0.022],
            [0.027, 0.027, 0.027, 0.027, 0.027, 0.027, 0.027, 0.027, 0.027, 0.027, 0.568, 0.135, 0.027],
            [0.048, 0.048, 0.048, 0.048, 0.048, 0.048, 0.048, 0.048, 0.048, 0.048, 0.048, 0.238, 0.238],
            [0.059, 0.059, 0.059, 0.059, 0.059, 0.059, 0.059, 0.059, 0.059, 0.059, 0.059, 0.294, 0.059]
        ])

        # labels order [0,1,2,3,4,5,6,7,8,9,10,11,12,-1,-2]
        self.camera_probab_table = np.array([
            [0.967, 0.004, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.006, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002],
            [0.019, 0.950, 0.001, 0.002, 0.004, 0.001, 0.001, 0.002, 0.001, 0.001, 0.012, 0.004, 0.001, 0.002, 0.001],
            [0.001, 0.004, 0.972, 0.000, 0.001, 0.009, 0.005, 0.000, 0.000, 0.001, 0.000, 0.002, 0.000, 0.000, 0.000],
            [0.001, 0.001, 0.001, 0.984, 0.001, 0.003, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
            [0.002, 0.016, 0.001, 0.001, 0.963, 0.006, 0.001, 0.001, 0.001, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001],
            [0.007, 0.001, 0.001, 0.003, 0.001, 0.934, 0.042, 0.001, 0.001, 0.005, 0.001, 0.001, 0.001, 0.001, 0.001],
            [0.002, 0.002, 0.001, 0.005, 0.001, 0.002, 0.981, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
            [0.002, 0.013, 0.001, 0.010, 0.003, 0.001, 0.001, 0.960, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
            [0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.938, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004],
            [0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.934, 0.005, 0.005, 0.005, 0.005, 0.005],
            [0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.883, 0.008, 0.008, 0.008, 0.008],
            [0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.023, 0.828, 0.011, 0.011, 0.011],
            [0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.835, 0.012, 0.012],
            [0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.928, 0.005],
            [0.020, 0.020, 0.020, 0.020, 0.020, 0.020, 0.020, 0.020, 0.020, 0.020, 0.020, 0.020, 0.020, 0.020, 0.720]
        ])

        """
        # use matrix for 9 floor buildings, old probability result

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
        """

    def countProbability(self):
        """
        creates bayesian network for data fusion

        # general inputs
        self.acc_diff_floor - measurement from accelerometer
        self.camera_floor - measurement from camera
        """
        self.bel_new = []
        # probability table starts from 0
        camera_floor = self.camera_floor
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
        self.bel = np.copy(np.round(self.bel_new[:], 2))

        # save bel to the array
        self.floor_probability_result.append(self.bel)

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

        return np.copy(self.bel)

    def plotProbability(self):
        """
        function plot probability belives for given floors during 5 steps of ride
        """
        plt.figure()
        for attempt in [0, 1, 2, 3, 4]:
            plt.plot(self.floor_probability_result[attempt])
        plt.title('Belive that robot is in given floor')
        plt.show()
