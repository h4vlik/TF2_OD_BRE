from main.flags_global import FLAGS
from input_feed import InputFeed
from core.accelerometer_estimation.acc_estimate_floor import detectFloorAcc
from core.camera_estimation.camera_estimate_floor import cameraEstimationFloor
from core.bayes_filter.bayesFilter import BayesFilter
from core.utils.utilsClass import utilsClass

import matplotlib.pyplot as plt
import numpy as np


class FloorEstimation(object):
    def __init__(self):
        self.inputFeed = InputFeed.InputFeed()
        self.detectFloorAcc = detectFloorAcc()
        self.cameraEstimationFloor = cameraEstimationFloor()
        self.bayesFilter = BayesFilter()
        self.utilsClass = utilsClass()
        self.RUN_CODE = True

        # information if elevator ride stop or not, True = stop, from acc_estimate_floor
        self.ride_stop = self.detectFloorAcc.elevators_ride_stop

        # inicialize estimated floor values
        self.time = 0
        self.diff_floor_acc = 0
        self.floor_camera = 0
        self.camera_frames = 2
        self.fusion_result = 0

        # loop iterator
        self.main_loop_iterator = 0

        # other init values
        self.camera_fps = 15
        self.acc_frequency = 100
        self.frame_divider = round(self.acc_frequency/self.camera_fps)

    def floor_estimation(self):
        """
        main loop
        """
        try:
            while(self.RUN_CODE):
                if FLAGS.detection_mode == 'online':
                    self.RUN_CODE = True
                elif FLAGS.detection_mode == 'offline':
                    if self.main_loop_iterator < self.inputFeed.end_count - 1:
                        self.RUN_CODE = True
                    else:
                        self.RUN_CODE = False

                acc_data = self.inputFeed.get_acc_data(self.main_loop_iterator)
                self.time = acc_data[0]
                self.diff_floor_acc = self.detectFloorAcc.spin(acc_data)

                # set floor_camera_array to empty array
                floor_camera_array = []

                # detect only when elevator stops
                if self.detectFloorAcc.elevators_ride_stop:
                    # LOOP OVER 3 NEXT FRAMES AND RETURN THE MOST FREQUENTED VALUE
                    for i in range(self.camera_frames):
                        frame = self.inputFeed.get_camera_data()
                        floor_camera = self.cameraEstimationFloor.spin(frame)

                        if floor_camera == 1000:
                            # bad classificaiton occured --> set camera result as fusion_result(t-1) + acc_diff_floor
                            floor_camera = int(np.copy(self.fusion_result) + np.copy(self.diff_floor_acc))
                            print("bad classification detected")
                        else:
                            pass

                        floor_camera_array = np.append(floor_camera_array, floor_camera)
                    print("Detections from camera: ", floor_camera_array)
                    self.floor_camera = self.utilsClass.getMostCommon(floor_camera_array)

                    print([self.floor_camera, self.diff_floor_acc])

                    # BAYES FILTER PART
                    bel_result = self.bayesFilter.spin(self.diff_floor_acc, self.floor_camera)
                    self.fusion_result = self.bayesFilter.getFloorResult()

                    # SAVE ACTUAL RESULT TO CSV FILE
                    self.utilsClass.saveData(self.time, self.floor_camera, self.diff_floor_acc, bel_result, self.fusion_result)

                else:
                    # load frame every Xth iteration, when NO ELEV STOP IS DETECTED
                    if (self.main_loop_iterator + 1) % self.frame_divider == 0 and FLAGS.detection_mode == 'offline':
                        self.inputFeed.CameraDataSource.get_next_frame()

                self.utilsClass.saveArrayData(self.time, self.floor_camera, self.diff_floor_acc, self.fusion_result)
                self.main_loop_iterator += 1  # iterate main loop iterator

        except KeyboardInterrupt:
            print("Program will be stopped ...")
            pass

        self.detectFloorAcc.plotData()
        self.utilsClass.visualizeRideFusion()
        self.bayesFilter.plotProbability()
