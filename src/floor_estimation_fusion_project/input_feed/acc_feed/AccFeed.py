"""Abstract Class AccFeed.
This module contains a wrapper class around ImageInput providers.
"""
from abc import ABC, abstractmethod


class AccFeed(ABC):
    @abstractmethod
    def get_acceleration(self):
        pass

    @abstractmethod
    def get_time(self):
        pass


class AccFeedEmpty(AccFeed):
    def get_acceleration(self):
        pass

    def get_time(self):
        pass
