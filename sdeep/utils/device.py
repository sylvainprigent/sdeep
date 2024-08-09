"""Singleton to access the device from all modules"""
import torch


def device() -> str:
    """Function for easier access to the device"""
    return Device.get()


class Device:
    """Singleton to access the device

    get the device using: Device.get()
    """
    __device = None

    def __init__(self):
        Device.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get() -> str:
        """ Static access method to the Config. """
        if Device.__device is None:
            Device()
        return Device.__device

    @staticmethod
    def print():
        """Print the device"""
        print("Device:", Device.__device)
