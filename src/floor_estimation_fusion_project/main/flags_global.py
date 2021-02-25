"""
Global variables for all scripts and parts of project.
"""

from absl import flags
import os
from pathlib import Path

# Flag names are globally defined!  So in general, we need to be
# careful to pick names that are unlikely to be used by other libraries.
# If there is a conflict, we'll get an error at import time.
"""
flags.DEFINE_string('name', 'Jane Random', 'Your name.')
flags.DEFINE_integer('age', None, 'Your age in years.', lower_bound=0)
flags.DEFINE_boolean('debug', False, 'Produces debugging output.')
flags.DEFINE_enum('job', 'running', ['running', 'stopped'], 'Job status.')
"""

flags.DEFINE_string('f', '', 'kernel')  # workaround due to JupyterNotebook

#####
# define main directory path
#####

flags.DEFINE_string(
    'main_dir_path',
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    'PATH to main folder path (TF_OD_BRE folder)')

# ===== ACC FEED FLAGS ====== #
flags.DEFINE_string('csv_input_file_path',
                    os.path.join(r'data\\Acc_input_data\\A3_01', os.path.basename("A3_01.csv")),
                    'PATH to a csv input file, where meassurement data are stored.')

flags.DEFINE_enum('detection_mode', 'offline',
                  ['online', 'offline'], 'Switch between online (realtime) and offline (from file) mode')

flags.DEFINE_string('csv_output_file_path',
                    os.path.join(r'results\\floor_detection_results', os.path.basename("Floor_detection_ACC.csv")),
                    'PATH to file, where all data from ride are stored')

flags.DEFINE_string('acc_device', 'COM3', 'for windows COM ports, for linux /dev/USB0')


# ===== CAMERA FEED FLAGS ======= #
# common
# flags.DEFINE_enum('image_input_mode', 'folder', ['camera', 'video', 'folder'], 'Source of image data.')

# image_feed_camera
flags.DEFINE_integer('camera_device_used', 0, 'An index of the used camera device', lower_bound=0)

# image_feed_video
flags.DEFINE_string('image_input_video_path',
                    os.path.join(r'data\Camera_input_data\video', os.path.basename("A3_01.mp4")),
                    'PATH to the folder with images, images passed recursively.')

# ===== CNN FLAGS ====== #
# image_feed_cvideo
flags.DEFINE_string('best_weights_path',
                    (r'results\\training_checkpoints\\colab_weigts_fine_tuning\\'),
                    'PATH to the folder with images, images passed recursively.')

# image_feed_camera
flags.DEFINE_integer('num_classes', 10, 'Number of classification classes', lower_bound=0)

# ===== FLAGS STOP ===== #


FLAGS = flags.FLAGS
