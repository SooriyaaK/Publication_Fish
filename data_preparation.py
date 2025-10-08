"""
Prepares the data from the original data set from Lei et al. (2020) and saves it as a csv 
ready for the experiments.

Run this before attempting any experiments as the required data file will otherwise be missing.
"""

from data_utils.data_preparation import DataPreparator
from enums.datasets_enum import Dataset

# prepare Lei et al. (2020) data with round fish tank and 5 fish
save_path_lei2020 = "data/lei2020_augmented_data.csv"
dataprepper_lei2020 = DataPreparator(Dataset.LEI2020, save_path=save_path_lei2020)
dataprepper_lei2020.prepare_data()



