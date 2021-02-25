"""
scripts define all data inputs needed for final algorithm

acceleration
time
image
"""
from input_feed.acc_feed import AccFeed, acc_realtime_feed, csv_file_feed
from input_feed.image_feed import ImageFeed, video_feed, camera_feed
from main.flags_global import FLAGS


class InputFeed(object):
    def __init__(self):
        self.AccData = [0, 0]
        self.AccDataSource = AccFeed.AccFeedEmpty()
        self.CameraDataSource = ImageFeed.ImageFeedEmpty()
        self.__assign_sources()

    # noinspection PyTypeChecker
    def __assign_sources(self):
        if FLAGS.detection_mode == 'online':
            self.AccDataSource = acc_realtime_feed.AccRealTimeFeed()
            self.CameraDataSource = camera_feed.CameraFeedAsync()
            self.end_count = 0
        elif FLAGS.detection_mode == 'offline':
            self.AccDataSource = csv_file_feed.CsvFileFeed()
            self.CameraDataSource = video_feed.VideoFeedAsync()
            self.end_count = self.AccDataSource.end_count
        else:
            self.AccDataSource = None
            self.CameraDataSource = None

        if self.AccDataSource is None or self.CameraDataSource is None:
            raise ValueError(f"detection_mode has not been assigned. Check FLAGS.")

        self.data_sources_assigned = True

    def get_next_input_data(self, main_loop_iterator):
        if self.data_sources_assigned is False:
            raise AttributeError("DataSources have not been assigned")
        self.AccData[0] = self.AccDataSource.get_time(main_loop_iterator)
        self.AccData[1] = self.AccDataSource.get_acceleration(main_loop_iterator)
        self.validate_input_data(ignore_validation=True)  # TODO: DEBUG ONLY !

    def validate_input_data(self, ignore_validation=False):
        if ignore_validation is True:
            return True
        for key, value in self.Data.items():
            if value is None:
                raise ValueError(f"{key} next value is None. Input feed interrupted. ")
            # TODO: more checks
        return True
        # TODO: raise errors if some data is invalid

    def get_acc_data(self, main_loop_iterator):
        self.get_next_input_data(main_loop_iterator)
        acc_data = self.AccData.copy()
        return acc_data

    def get_camera_data(self):
        self.frame = self.CameraDataSource.get_frame()
        self.CameraDataSource.show_frame()
        frame = self.frame.copy()
        return frame
