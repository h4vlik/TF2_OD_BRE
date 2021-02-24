"""
load acc and time from .csv file
"""
from input_feed.acc_feed.AccFeed import AccFeed
from main.flags_global import FLAGS

import numpy as np
import csv
import os


class CsvFileFeed(AccFeed):
    def __init__(self):
        self.main_dir = FLAGS.main_dir_path
        self.csv_file_path = os.path.join(self.main_dir, FLAGS.csv_input_file_path)
        self.acc = 0
        self.__loadCSV()

    def __loadCSV(self):
        """
        load data from .csv file in array
        """
        my_data = np.genfromtxt(
            self.csv_file_path,
            delimiter='\t')

        # data z akcelerometru
        self.load_time_array = my_data[1:, 0]
        self.load_acc_array = my_data[1:, 1]

        # get length of input data
        self.end_count = self.load_time_array.shape[0]
        print("delka vstupnich dat: ", self.end_count)

    def get_acceleration(self, main_loop_iterator):
        """
        load data from .csv file like from sensor, one value for index
        """
        self.acc = self.load_acc_array[main_loop_iterator]

        return self.acc.copy()

    def get_time(self, main_loop_iterator):
        """
        load time data from .csv file like from sensor
        """
        self.time = self.load_time_array[main_loop_iterator]

        return self.time.copy()
