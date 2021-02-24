"""
scripts with detectFloorAcc class, that provides all functions connected with floor detection via accelerometer
"""
from input_feed import InputFeed

import time
import datetime
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy import signal, fft, integrate
import pandas
import os
import json


# script for offline floor detection based on real-time-floor detection
class detectFloorAcc(object):
    def __init__(self):
        self.input = InputFeed.InputFeed()
        self.end_count = self.input.end_count

        # Initialization of communication
        self.sample_freq = 100  # frequency of input data
        self.delta_time = 1/self.sample_freq  # sample time of input data

        # Parameters - constants during running the program
        self.deltaVelMax = 0.01  # this is number after ride one floor - calibration - NEED TO BE DONE
        self.cal_loop_count = self.sample_freq*13  # 10 seconds to ride one floor up or down, for elevator calibration
        self.deltaVelMaxMultiplier_low = 0.01
        self.halfMaxVel = 0.2
        self.deltaVelMaxMultiplier_positive = 0.3
        self.deltaVelMaxMultiplier_negative = 0.2
        self.length_floor = 2.35  # floor lenght in meters
        self.acc_low_treshold = 0.1

        # Inicialization
        self.integrate = False  # boolean
        self.calibration_not_completed = True  # boolean
        self.state = "ride calibration"  # variable for state machine

        # buffer inicialization
        self.buffer_length = 50  # sample count for filtration buffer
        self.filt_buffer = []  # buffer for filtration
        self.ACC_ARRAY_FILT = []  # array to save all data during code run

        # init filters
        self.__init_filters()

        # array for plot inicialization
        self.acc_array = []
        self.delta_vel_array = []
        self.velocity_array = []
        self.position_array = []
        self.time_array = []
        self.floor_array = []
        self.floor_vector = []

        # other inicializations
        self.main_loop_iterator = 0  # variable for counting iterations in the main loop
        self.loop_iterator = 0  # variable for counting iterations
        self.buffer_iterator = 0  # variable for counting buffer iterations
        self.line = 0
        self.acc = 0
        self.acc_prev = 0
        self.time = 0
        self.time_prev = 0
        self.offset = 0
        self.vel = 0
        self.vel_prev = 0
        self.delta_acc = 0
        self.delta_vel = 0
        self.pos_prev = 0
        self.pos = 0
        self.floor_number = 0
        self.floor_number_prev = 0
        self.floor_number_eval = 0

        # starting position
        self.starting_floor = 0

        # elevator stop, True = elevators ride is over, False = elevator rides
        self.elevators_ride_stop = False

    def __init_filters(self):
        """
        create filters
        """
        # filter inicialization
        # Bandpass filter
        highpass_cutoff = 0.01  # filter DC parts of signal
        lowpass_cutoff = 10  # filter noise
        self.bandpass = signal.butter(
            2,
            [2*highpass_cutoff/self.sample_freq, 2*lowpass_cutoff/self.sample_freq],
            'bandpass',
            analog=False,
            output='ba'
            )
        # Lowpass filter for noise elimination, 10 Hz cutoff
        self.DLPF = signal.butter(
            1,
            2*lowpass_cutoff/self.sample_freq,
            'lowpass',
            analog=False,
            output='ba')

    def calibrateRide(self):
        """
        calibrate tresholds for floor estimation part when elevator rides up
        input: self.button_pressed --> information if button is pressed (True, False)
        input: self.acc --> acceleration
        input: self.display_digit --> what is in the display ??

        output: self.deltaVelMaxUp
        output: self.halfMaxVel
        output:
        """

        i = 0

        if self.main_loop_iterator == 0:
            print("Move elevator up. Calibration is running ... ")
            self.time_array = np.append(self.time_array, self.time)
            self.acc_array = np.append(self.acc_array, self.acc)

        elif self.main_loop_iterator > self.cal_loop_count and abs(self.acc) < self.acc_low_treshold:
            # filter
            self.acc_array = signal.filtfilt(self.DLPF[0], self.DLPF[1], self.acc_array)
            # integrate to velocity
            self.velocity_array = self.delta_time * integrate.cumtrapz(self.acc_array, initial=0)
            # count max vel
            self.halfMaxVel = 0.5*np.amax(np.absolute(self.velocity_array))
            # count delta vel - difference
            self.delta_vel_array = np.diff(self.velocity_array, prepend=[0])
            self.deltaVelMax = np.amax(np.absolute(self.delta_vel_array))
            self.integrate = False

            velocity_array = [0]

            for i in range(len(self.delta_vel_array)-1):
                self.time_prev = self.time_array[i]
                self.time = self.time_array[i+1]
                self.acc = self.acc_array[i+1]
                self.acc_prev = self.acc_array[i]
                self.getVelocity()

                if self.integrate:
                    if abs(self.delta_vel) < (self.deltaVelMax*self.deltaVelMaxMultiplier_low):
                        if abs(self.vel) > self.halfMaxVel:
                            self.vel = self.vel_prev
                            self.integrate = False
                        else:
                            self.vel = 0
                            self.vel_prev = 0
                            self.integrate = False
                            self.elevators_ride_stop = True
                else:
                    # pro kladnou deltu
                    if self.delta_vel >= (self.deltaVelMax*self.deltaVelMaxMultiplier_positive):
                        # zacni integrovat
                        self.integrate = True
                        self.elevators_ride_stop = False

                    # pro zapornou deltu
                    elif abs(self.delta_vel) >= (self.deltaVelMax*self.deltaVelMaxMultiplier_negative) and self.delta_vel < 0:
                        self.integrate = True
                        self.elevators_ride_stop = False

                    else:
                        # neintegruju, zachovam predeslou hodnotu
                        self.vel = self.vel_prev
                velocity_array = np.append(velocity_array, self.vel)
                self.vel_prev = self.vel

            # integrate to position
            self.position_array = self.delta_time * integrate.cumtrapz(velocity_array, initial=0)

            # count tresholds
            print("half of max velocity: ", self.halfMaxVel)
            print("velocity difference - max value: ", self.deltaVelMax)
            self.length_floor = np.amax(np.absolute(self.position_array))
            print("delka jednoho patra: ", self.length_floor)

            f, axs = plt.subplots(4, 1, sharey=True)

            axs[0].plot(self.time_array, self.acc_array)
            axs[0].set_ylabel('acc [m/ss]')

            axs[1].plot(self.time_array, velocity_array)
            axs[1].set_ylabel('vel [m/s]')

            axs[2].plot(self.time_array, self.position_array)
            axs[2].set_ylabel('position ride Up [m]')

            axs[3].plot(self.time_array, self.delta_vel_array)
            axs[3].set_ylabel('delta vel [m]')
            plt.show()

            # clear all data
            self.time_array = []
            self.velocity_array = []
            self.delta_vel_array = []
            self.acc_array = []
            self.position_array = []
            self.loop_iterator = 0
            self.acc = 0
            self.acc_prev = 0
            self.vel = 0
            self.vel_prev = 0
            self.delta_vel = 0
            self.pos = 0
            self.pos_prev = 0
            self.floor_number = 0
            self.time = 0
            self.time_prev = 0

            self.state = "estimate floor"  # switch to next state
            print(" <<<<<< Kalibrace pro jizdu nahoru hotova >>>>>> \n")
            print("----- FLOOR ESTIMATION RUNNING ------\n")

        else:
            self.time_array = np.append(self.time_array, self.time)
            self.acc_array = np.append(self.acc_array, self.acc)

    def loadDataToBuffer(self):
        # load data to buffer
        self.filt_buffer = np.append(self.filt_buffer, self.acc)

        if self.buffer_iterator == (self.buffer_length-1):
            # filter buffer data
            self.filterData()

            # save data from buffer to acc_array
            self.ACC_ARRAY_FILT = np.append(self.ACC_ARRAY_FILT, self.filt_buffer)

            # clear buffer
            self.filt_buffer = []
            self.buffer_iterator = 0
        else:
            self.buffer_iterator += 1

    def filterData(self):
        # bandpass filter
        # self.filt_buffer = signal.filtfilt(self.bandpass[0], self.bandpass[1], self.filt_buffer)

        # low pass filter - DLPF
        self.filt_buffer = signal.filtfilt(self.DLPF[0], self.DLPF[1], self.filt_buffer)

    # count raw velocity by integration, count delta_vel, dont update
    def getVelocity(self):
        # derivace
        # vypocitam deltu
        self.delta_time = self.time - self.time_prev
        self.delta_acc = self.acc - self.acc_prev

        # vypocitam rychlost (integruji)
        self.vel = (self.acc_prev + self.delta_acc) * self.delta_time
        self.vel = self.vel + self.vel_prev

        # vypocitam deltu vel
        self.delta_vel = self.vel - self.vel_prev

    # vypocita pomoci integrace polohu
    def getPosition(self):
        # derivace
        # vypocitam polohu (integruji)
        self.pos = np.round((self.vel_prev + self.delta_vel) * self.delta_time, 2)
        self.pos = self.pos + self.pos_prev

        # print("Aktualni pozice: ",self.pos)

        # vypocitam pozici - pokud se pozice v dalsim kroku nezmeni --> cekam v patre --> zaokrouhlim na cela patra
        # kdyz se zacne menit, pricitam uz k poloze, ktera odpovida konkretnimu patru --> zmensim chybu
        # UPRAVENO - NUTNE OVERIT
        if self.pos == self.pos_prev and self.vel == 0:
            self.floor_number = np.round((self.pos/self.length_floor), 0)
            # count probability
            # self.countProbability()
            self.pos = self.floor_number * self.length_floor
        else:
            self.floor_number = np.round((self.pos/self.length_floor), 0)

        # count diff floor and append it to floor vector
        self.countVectorFloor()

        # actualize data
        self.pos_prev = np.copy(self.pos)
        self.floor_number_prev = np.copy(self.floor_number_eval)

    def countProbability(self):
        prob_array = []
        for i in range(8):
            prob = (1 - abs(np.round((i - self.pos/self.length_floor), 2)))
            if prob < 0:
                prob = 0
            prob_array.append(prob)
        print("probability of well estimated floor is: ", prob_array, "time:", self.time)

    def countVectorFloor(self):
        """
        count difference of floor number and if the difference is not 0, write it to the floor vector

        output: floor_vector
        """
        floor_diff = self.floor_number_eval - self.floor_number_prev

        if floor_diff == 0:
            pass
        else:
            self.floor_vector = np.append(self.floor_vector, floor_diff)

    def evaluateDetection(self):
        """
        compare real floor vector with computed floor vector and write it to the /res/evaluation_results.csv
        """
        # vytvorime soubor pro ukladani vysledku
        path_to_res_file = os.path.join(
            self.main_dir_path,
            r'res\\',
            os.path.basename("evaluation_results.csv"))

        file = open(path_to_res_file, "a")

        for i, floor in enumerate(self.floor_vector):
            if self.real_floor_vector[i] == self.floor_vector[i]:
                result = "true"
            else:
                result = "false"

            # save data to .txt
            str_data = str(str(self.real_floor_vector[i])+'\t'+str(self.floor_vector[i])+'\t'+str(result))
            file.write(str_data + "\n")  # write data with a newline

        print("vypocitany vektor podlazi: ", self.floor_vector)
        print("realny vektor podlazi: ", self.real_floor_vector)

    def saveData(self):
        # save data to array
        self.acc_array = np.append(self.acc_array, self.acc)
        self.velocity_array = np.append(self.velocity_array, self.vel)
        self.position_array = np.append(self.position_array, self.pos)
        self.time_array = np.append(self.time_array, self.time)
        self.floor_array = np.append(self.floor_array, self.floor_number)

    def plotData(self):
        _, axs1 = plt.subplots(nrows=4, ncols=1, figsize=(10, 6))
        rows = ['{}'.format(row) for row in ['filt_acceleration \n[m/ss]', 'velocity \n[m/s]', 'position \n[m]', 'floor \n[-]']]

        for ax, row in zip(axs1, rows):
            ax.set_ylabel(row)
            ax.grid()

        axs1[0].plot(self.time_array, self.acc_array)
        axs1[1].plot(self.time_array, self.velocity_array)
        axs1[2].plot(self.time_array, self.position_array)
        axs1[3].plot(self.time_array, self.floor_array)
        axs1[3].set_xlabel("time [s]")

    def spin(self):
        # MAIN LOOP
        acc_data = self.input.get_acc_data(self.main_loop_iterator)
        self.acc = acc_data[1]
        self.time = acc_data[0]

        # filtration part
        self.loadDataToBuffer()
        if self.main_loop_iterator > (self.buffer_length-1):
            index = self.main_loop_iterator - self.buffer_length
            self.acc = self.ACC_ARRAY_FILT[index]
        else:
            # first X values (according to buffer length) is 0
            self.acc = 0

        # calibrate elevator
        if self.state == "ride calibration":
            self.calibrateRide()

        # calibration is completed --> start estimate the floor number
        elif self.state == "estimate floor":

            # count delta_vel
            self.getVelocity()

            if self.integrate:
                if abs(self.delta_vel) < (self.deltaVelMax*self.deltaVelMaxMultiplier_low):
                    if abs(self.vel) > self.halfMaxVel:
                        self.vel = self.vel_prev
                        self.integrate = False
                    else:
                        # part where integration stops --> SAVE FLOOR NUMBER AND LOAD IMAGE TO CLASSIFY IT
                        self.vel = 0
                        self.vel_prev = 0
                        self.acc = 0
                        self.integrate = False
                        self.floor_number_eval = np.copy(self.floor_number)
            else:
                # pro kladnou deltu
                if self.delta_vel >= (self.deltaVelMax*self.deltaVelMaxMultiplier_positive):
                    # zacni integrovat
                    self.integrate = True

                # pro zapornou deltu
                elif abs(self.delta_vel) >= (self.deltaVelMax*self.deltaVelMaxMultiplier_negative) and self.delta_vel < 0:
                    self.integrate = True

                else:
                    # neintegruju, zachovam predeslou hodnotu
                    self.vel = self.vel_prev

            self.getPosition()
            self.saveData()

        # aktualizuji data
        self.vel_prev = np.copy(self.vel)
        self.acc_prev = np.copy(self.acc)
        self.time_prev = np.copy(self.time)
        self.main_loop_iterator += 1  # iterate main loop iterator


def createResultsFile(main_dir_path):
    """
    create csv file for saving results
    """
    # vytvorime soubor pro ukladani vysledku
    path_to_res_file = os.path.join(
        main_dir_path,
        r'res\\',
        os.path.basename("evaluation_results.csv"))
    header = str("real floor"+"\t"+"count floor"+"\t"+"result"+"\n")
    file = open(path_to_res_file, "a")
    file.write(header)  # write data with a newline


def load_real_data(folder, filename, json_data):
    """
    load information about real floor from json file
    count real floor vector (difference of real floor information)
    """
    real_floor_vector = np.diff(json_data[folder][filename]["floor"])
    print(real_floor_vector)

    return real_floor_vector


# --- kod, ktery se provede --- #
def main():
    # define path
    main_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # breach_dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # create file for saving results
    createResultsFile(main_dir_path)

    # path to csv directory
    path_to_csv_dir = os.path.join(
        main_dir_path,
        r'Data_for_fusion\\MERENI_18_2_2021_strojarna\\')

    # path to real floor data json file
    path_to_json_file = os.path.join(
        main_dir_path,
        r'Data_for_fusion',
        os.path.basename("mereni_18_2_2021_real_floor.json"))

    # open json file
    f = open(path_to_json_file)
    real_floor_data = json.load(f)

    # loop over all folders in csv_soubory
    for _, folder in enumerate(os.listdir(path_to_csv_dir)):
        print("new folder:", folder)
        # loop over all files in folder
        for _, filename in enumerate(os.listdir(path_to_csv_dir+folder)):
            print("new file:", filename)

            # path to .csv for evaluation
            path_to_csv_file = os.path.join(
                path_to_csv_dir,
                folder+'\\',
                os.path.basename(filename))
            # path to csv where calibration data are stored
            path_to_calibration_csv_file = os.path.join(
                path_to_csv_dir,
                folder+'\\',
                os.path.basename("kalibrace.csv"))

            # loaded real floor vector for evaluation
            real_floor_vector = load_real_data(folder, filename, real_floor_data)

            # main part of code
            detect_floor = detectFloorAcc(path_to_csv_file, path_to_calibration_csv_file, real_floor_vector)
            detect_floor.spin()
            detect_floor.evaluateDetection()
            detect_floor.plotData()
            plt.show()


if __name__ == '__main__':
    main()
