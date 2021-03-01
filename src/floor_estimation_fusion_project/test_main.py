from main.flags_global import FLAGS
from input_feed import InputFeed
from core.accelerometer_estimation.acc_estimate_floor import detectFloorAcc
from core.camera_estimation.camera_estimate_floor import cameraEstimationFloor

import matplotlib.pyplot as plt


class FloorEstimation(object):
    def __init__(self):
        self.inputFeed = InputFeed.InputFeed()
        self.detectFloorAcc = detectFloorAcc()
        self.cameraEstimationFloor = cameraEstimationFloor()
        self.RUN_CODE = True

        # information if elevator ride stop or not, True = stop, from acc_estimate_floor
        self.ride_stop = self.detectFloorAcc.elevators_ride_stop

        # inicialize estimated floor values
        self.floor_acc = 0
        self.floor_camera = 0

        # loop iterator
        self.main_loop_iterator = 0

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
                self.floor_acc = self.detectFloorAcc.spin(acc_data)

                # load frame every Xth iteration
                if (self.main_loop_iterator + 1) % self.frame_divider == 0:
                    self.inputFeed.CameraDataSource.get_next_frame()

                # detect only when elevator stops
                if self.detectFloorAcc.elevators_ride_stop:
                    frame = self.inputFeed.get_camera_data()
                    self.floor_camera = self.cameraEstimationFloor.spin(frame)
                    self.detectFloorAcc.elevators_ride_stop = False
                    print([self.floor_camera, self.floor_acc])
                self.main_loop_iterator += 1  # iterate main loop iterator

        except KeyboardInterrupt:
            print("Program will be stopped ...")
            pass

        self.detectFloorAcc.plotData()
        plt.show()
