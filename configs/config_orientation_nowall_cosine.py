import string
from dataclasses import dataclass, field
from enums.datasets_enum import Dataset
from enums.loss_functions_enum import LossFunctions
from enums.sorting_criteria_enum import SortingCriteria
from enums.wall_behaviour_enum import WallBehaviourType

from configs.config import Config

@dataclass
class ConfigOrientationNoWallCosine(Config):
    # --------------------------------------------------------------
    # GENERAL
    # --------------------------------------------------------------
    name: string = "lei2020_orientation_without_wall_cosine" # the name - used for file name generation. Ideally unique
    iterations: int = 50 # number of iterations for the experiment

    # --------------------------------------------------------------
    # DATA
    # --------------------------------------------------------------
    data_file_path: string = "data/lei2020_augmented_data.csv" # where the data has been stored
    dataset: Dataset = Dataset.LEI2020 # the dataset used for the evaluation
    experiment_ids: list = field(default_factory=lambda: [151]) # which experiments should be used for evaluation
    fish_ids: list = field(default_factory=lambda: [3]) # which fish should be used for every experiment stated in experiment_ids

    # --------------------------------------------------------------
    # SETTINGS FOR COMPARISONS
    # --------------------------------------------------------------
    sorting_criteria: SortingCriteria = SortingCriteria.ORIENTATION # how to sort the neighbours
    wall_behaviour: WallBehaviourType = WallBehaviourType.NO_WALL # how the focal fish reacts to the presence of a wall
    loss_function: LossFunctions = LossFunctions.COSINE # how the loss/fitness should be computed in the EA
  
    # --------------------------------------------------------------
    # EA HYPERPARAMS
    # --------------------------------------------------------------
    num_generations: int = 100 # number of generations
    population_size: int = 40 # population size (number of individuals)
    p_crossover: float = 0.6 # crossover probability
    mutation_rate: float = 0.2 # mutation rate
    k: int = 8 # number of individuals selected for tournament
    dimensions: int = 6 # size of the weights vector
