"""
Global variables for all scripts and parts of project.
"""

from absl import flags
import os

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

#####
# CAMERA FLAGS START
# common
flags.DEFINE_enum('image_input_mode', 'folder', ['camera', 'video', 'folder'], 'Source of image data.')

# image_feed_camera
flags.DEFINE_integer('camera_device_used', 0, 'An index of the used camera device', lower_bound=0)
# image_feed_camera
flags.DEFINE_string('image_input_folder_path', r"C:\Users\cernil\OneDrive - Y Soft Corporation "
                                               r"a.s\DeepLearningBlackbox\test\test_images_elevator_control_panel",
                    'PATH to the folder with images, images passed recursively.')
# image_feed_video
flags.DEFINE_string('image_input_video_path', None, 'PATH to the folder with video, used while image_input_mode=video.')


# INPUT FEED FLAGS END
#####
# ELEVATOR CONTROLS DETECTION FLAGS START
# elevator element detection
flags.DEFINE_string('detector_elements_model_path', r"C:\Users\cernil\OneDrive - Y Soft Corporation "
                                                    r"a.s\DeepLearningBlackbox\test\output\saved_model",
                    'PATH to a SavedModel file capable of detection of elevator elements.')
flags.DEFINE_enum('detection_elements_model_type', 'tf2',
                  ['tf2', 'reserved'], 'Type of detection model used - RESERVED.')
flags.DEFINE_string('label_map_path_detection',
                    r"C:\Users\cernil\OneDrive - Y Soft Corporation "
                    r"a.s\DeepLearningBlackbox\test\output\pascal_label_map.pbtxt",
                    'PATH to the label_map.txt | label_map.pbtxt file for detection.')
# floor button classification
flags.DEFINE_string('classification_floor_button_model_path', r"C:\Users\cernil\OneDrive - Y Soft Corporation "
                                                              r"a.s\DeepLearningBlackbox\button_classifier",
                    'PATH to a SavedModel file capable of classification of elevator floor buttons.')
flags.DEFINE_enum('classification_floor_button_model_type', 'keras',
                  ['keras', 'reserved'], 'Type of classification model used - RESERVED.')
# ELEVATOR CONTROLS DETECTION FLAGS END
#####


FLAGS = flags.FLAGS
