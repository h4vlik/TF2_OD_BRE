"""
class that provides various functions
"""
from main.flags_global import FLAGS

import time
import datetime
import csv
import json
import statistics
import numpy as np
import matplotlib.pyplot as plt


class utilsClass(object):
    """
    class of all utils neccessary for program

    saveData - saves result of data fusion to .csv file
    getMostCommon - returns most common number from list
    """
    def __init__(self):
        # create arrays for saving data from fusion
        self.time_array = []
        self.acc_floor_array = []
        self.camera_floor_array = []
        self.fusion_floor_array = []
        # create file .csv for saving data
        self.__createFileHeader()

    def __createFileHeader(self):
        # vytvorime soubor pro ukladani dat a zapisu tam header
        self.fileName = "results_data.csv"  # name of the CSV file generated
        header = str("time"+"\t"+"camera_floor"+"\t"+"acc_diff_floor"+"\t"+"FUSION RESULT"+"\t"+"bayes_floor"+"\n")
        file = open(self.fileName, "a")
        file.write(header)  # write data with a newline

        # vytvorime soubor pro ukladani dat a zapisu tam header
        self.file_name_2 = "fusion_floor_results_file_with_time.csv"  # name of the CSV file generated
        header = str("time"+"\t"+"Acc_floor"+"\t"+"Camera_floor"+"\t"+"Fusion_floor"+"\n")
        file = open(self.file_name_2, "a")
        file.write(header)  # write data with a newline

    def saveData(self, time, camera_floor, acc_diff_floor, bayes_bel, fusion_result):
        """
        save data from fusion to .csv
        """
        str_data = str(str(time)+'\t'+str(camera_floor)+'\t'+str(acc_diff_floor)+'\t'+str(fusion_result)+'\t'+str(bayes_bel))
        file = open(self.fileName, "a")
        file.write(str_data + "\n")  # write data with a newline

    def getMostCommon(self, floor_array):
        """
        returns most common number from list

        input: list of floor numbers from camera detection

        output: most common number of floor from detections
        """

        return int(statistics.mode(floor_array))

    def saveArrayData(self, time, camera_floor, acc_floor, fusion_floor):
        """
        saves all input to and output from fusion to arrays (acc_result, camera_result, fusion_result)
        """
        self.time_array = np.append(self.time_array, time)
        self.acc_floor_array = np.append(self.acc_floor_array, acc_floor)
        self.camera_floor_array = np.append(self.camera_floor_array, camera_floor)
        self.fusion_floor_array = np.append(self.fusion_floor_array, fusion_floor)

        str_data = str(str(time)+'\t'+str(acc_floor)+'\t'+str(camera_floor)+'\t'+str(fusion_floor))
        file = open(self.file_name_2, "a")
        file.write(str_data + "\n")  # write data with a newline

    def visualizeRideFusion(self):
        """
        visualize all fusion data from ride
        """
        # Major ticks every 20, minor ticks every 5
        major_ticks = np.arange(-12, 12, 1)

        _, axs1 = plt.subplots(nrows=3, ncols=1, figsize=(10, 6))
        rows = ['{}'.format(row) for row in ['floor diff - from acceleration \n[-]', 'floor from camera \n[-]', 'floor from fusion \n[-]']]

        for ax, row in zip(axs1, rows):
            ax.set_ylabel(row)
            ax.set_yticks(major_ticks)
            ax.grid()

        axs1[0].plot(self.time_array, self.acc_floor_array)
        axs1[1].plot(self.time_array, self.camera_floor_array)
        axs1[2].plot(self.time_array, self.fusion_floor_array)
        axs1[2].set_xlabel("time [s]")

        plt.show()
