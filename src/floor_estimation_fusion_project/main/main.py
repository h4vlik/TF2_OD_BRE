"""
main script of elevator floor detection project
"""

from floor_estimation_fusion_project.main.flags_global import FLAGS
from floor_estimation_fusion_project.main.flags_global import FLAGS

# 1. INPUT
#  from acc / from file, from camera / from video

# 2. Load Model or weights
#  model, model + weights

# 3. load data to bayes

# 4. show result

# main problem - how to synchronize video and acc, acc - 100 Hz, video 20 Hz,
# image classification - only when elevator stops


"""
functions:

calibrate_sensors() - calibrate acc sensor, count offset value
calibrate_ride_up() - calibrate for exact elevator (ride one floor up)

load_acc_data - sample freq 100 Hz
load_image_data - sample freq 20 Hz or load image only when elevator stops
load_model - load model + best weights

floor_detection_acc() - every loop count actual position and floor number
display_classification_camera() - when call, classify input image

bayes_fusion() - fuse actual floor from display number and from accelerometer output (diff_floor) 

correct_position() - according to bayes_fusion output correct position and floor number

stop_loop() - save actual position and stop floor_estimation_algorithm


#### PSEUDO CODE ####
load_model()
load_acc_data()

calibrate_sensor()
# state machine or if condition??
calibrate_ride_up()

LOOP:
# every iteration count actual position and floor via acc
floor_detection_acc()

# if elevator stops, classify display and fuse data
if elevator_status == stop:
    load_image_data()
    display_classification()
    bayes_fusion()
    correct_position()

if estimate_floor_number == goal_floor_number
    stop_loop()
    elevator_status = goal_positon

next iteration
"""


class FloorEstimation(object):
    def __init__(self):
        self.detection_mode = FLAGS.detection_mode
        self.serie = 0

    def load_model(self):
        """
        function load model of CNN with best weights
        """
    pass

    def CreateFileHeader(self):
        # vytvorime soubor pro ukladani dat a zapisu tam header
        self.fileName = "test_realtime_data.csv"  # name of the CSV file generated
        header = str("time"+"\t"+"acc_z"+"\t"+"vel_z"+"\t"+"pos_z"+"\t"+"floor_number"+"\n")
        file = open(self.fileName, "a")
        file.write(header)  # write data with a newline

    def startCommunication(self):
        # Initialiazation of communication
        self.serie = serial.Serial(self.com, self.speed)  # get serial data from acc

        # Wait for serial com to ready up
        print("Opening serial comms port...")
        time.sleep(5)

        # clear serial com buffer
        self.serie.reset_input_buffer()
        self.serie.reset_output_buffer()

        print("Comms and Sensors ready")

    # nastavi komunikaci s arduinem
    def getLine(self):
        self.line = str(self.serie.readline(), 'utf-8')
        self.line = self.line.split("\t")  # list of strings

    # zapise hodnotu zrychleni (v ose z) do floatu
    def makeAccLinVec(self):
        # from string to float and store to acc_vector
        if len(self.line) == 4:
            self.acc = float(self.line[2])
            self.acc = - self.acc - self.offset  # convert to good polarity
        else:
            self.acc = self.acc_prev
        # print("zrychleni je: ", self.acc)
        # save unfiltered data
        self.acc_raw = np.copy(self.acc)

    def load_acc_data(self):
        """
        function load acc data in each iteration

        1. from csv.
            - load csv file
            - according to loop iterator load acc and time data from array

        2. from sensor
            - start communication
            - calibrate sensor
        """
        if self.detection_mode == "online":
            # create file .csv and header - file where data are stored
            self.CreateFileHeader()

            # START COMMUNICATION
            while True:
                try:
                    self.startCommunication()
                except:
                    print("Connection error, try to connect device")
                    time.sleep(2)
                else:
                    break
