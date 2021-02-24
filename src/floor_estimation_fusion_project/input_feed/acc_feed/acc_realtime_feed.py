"""
"""
from input_feed.acc_feed.AccFeed import AccFeed
from main.flags_global import FLAGS

import numpy as np
import csv
import serial
import time


class AccRealTimeFeed(AccFeed):
    def __init__(self):
        self.main_dir = FLAGS.main_dir_path
        self.acc = 0
        self.acc_prev = 0
        self.time = 0
        self.time_prev = 0
        self.offset = 0

        # Setup
        # :TODO doplnit do global flags
        self.com = 'COM4'
        self.speed = 115200

        # Initialiazation of communication
        self.sample_freq = 100  # frequency of input data
        self.delta_time = 1/self.sample_freq  # sample time of input data
        self.serie = 0
        self.line = 0
        self.cal_loop_count = self.sample_freq*2  # calibrate senzor for 2 sec
        self.loop_iterator = 0

        # for calibration
        self.acc_array = []

        self.__startCommunication()
        self.__calibrateSensor()

    def __startCommunication(self):
        # START COMMUNICATION
        while True:
            try:
                # Initialiazation of communication
                self.serie = serial.Serial(self.com, self.speed)  # get serial data from acc

                # Wait for serial com to ready up
                print("Opening serial comms port...")
                time.sleep(5)

                # clear serial com buffer
                self.serie.reset_input_buffer()
                self.serie.reset_output_buffer()

                print("Comms and Sensors ready")
            except:
                print("Connection error, try to connect device")
                time.sleep(2)
            else:
                break

    def __calibrateSensor(self):
        """
        function for calibration of sensor
        --> measure for X seconds acceleration, count offset and subtract it from acc
        """
        while self.loop_iterator <= self.cal_loop_count:
            acc = self.get_acceleration(0)
            if self.loop_iterator == 0:
                # append acc to array
                self.acc_array = np.append(self.acc_array, acc)
                print("Calibration of sensor is running ...")

            elif self.loop_iterator == self.cal_loop_count:
                self.offset = np.mean(self.acc_array)  # count offset
                print(" Calibration is done!")
                print("Offset is: ", self.offset)

            else:
                # append acc to array
                self.acc_array = np.append(self.acc_array, acc)

            # actualize loop count
            self.loop_iterator += 1

    # nastavi komunikaci s arduinem
    def getLine(self):
        self.line = str(self.serie.readline(), 'utf-8')
        self.line = self.line.split("\t")  # list of strings

    # zapise hodnotu zrychleni (v ose z) do floatu
    def makeAccLinVec(self):
        # from string to float and store to acc_vector
        if len(self.line) == 4:
            acc = float(self.line[2])
            self.acc = - acc - self.offset  # convert to good polarity
        else:
            self.acc = self.acc_prev.copy()

    def savePrevAcc(self):
        self.acc_prev = np.copy(self.acc)

    def get_acceleration(self, main_loop_iterator):
        """
        load data from from sensor
        """
        self.getLine()
        self.makeAccLinVec()
        self.savePrevAcc()

        return self.acc

    def get_time(self, main_loop_iterator):
        """
        load time from sensor
        """
        self.time = self.time_prev + self.delta_time
        self.time_prev = np.copy(self.time)

        return self.time
