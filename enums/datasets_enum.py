from enum import Enum

"""
Enum of datasets
"""
class Dataset(Enum):
    LEI2020 = "lei2020", "data/lei2020_original.dat", ["exp_id", "fish_id", "kicktime", "x", "y"]

    def __init__(self, label, file_path, headers):
        self.label = label
        self.file_path = file_path
        self.headers = headers