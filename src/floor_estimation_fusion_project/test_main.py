from main.flags_global import FLAGS
from input_feed import InputFeed
from core.accelerometer_estimation.acc_estimate_floor import detectFloorAcc

import matplotlib.pyplot as plt


class FloorEstimation(object):
    def __init__(self):
        self.detectFloorAcc = detectFloorAcc()
        self.RUN_CODE = True
        self.ride_stop = self.detectFloorAcc.elevators_ride_stop

    def floor_estimation(self):
        """
        main loop
        """
        try:
            while(self.RUN_CODE):
                if FLAGS.detection_mode == 'online':
                    self.RUN_CODE = True
                elif FLAGS.detection_mode == 'offline':
                    if self.detectFloorAcc.main_loop_iterator < self.detectFloorAcc.end_count - 1:
                        self.RUN_CODE = True
                    else:
                        self.RUN_CODE = False

                self.detectFloorAcc.spin()
            self.detectFloorAcc.plotData()
            plt.show()

        except KeyboardInterrupt:
            print("Program will be stopped ...")
            pass