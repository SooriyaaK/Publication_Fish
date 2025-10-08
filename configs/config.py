import string
from dataclasses import dataclass, field
from enums.datasets_enum import Dataset
from enums.loss_functions_enum import LossFunctions
from enums.sorting_criteria_enum import SortingCriteria
from enums.wall_behaviour_enum import WallBehaviourType

@dataclass
class Config:
    # --------------------------------------------------------------
    # GENERAL
    # --------------------------------------------------------------
    name: string # the name - used for file name generation. Ideally unique
    iterations: int # number of iterations for the experiment

    # --------------------------------------------------------------
    # DATA
    # --------------------------------------------------------------
    data_file_path: string # where the data has been stored
    dataset: Dataset # the dataset used for the evaluation
    experiment_ids: list # which experiments should be used for evaluation
    fish_ids: list # which fish should be used for every experiment stated in experiment_ids

    # --------------------------------------------------------------
    # SETTINGS FOR COMPARISONS
    # --------------------------------------------------------------
    sorting_criteria: SortingCriteria # how to sort the neighbours
    wall_behaviour: WallBehaviourType # how the focal fish reacts to the presence of a wall
    loss_function: LossFunctions # how the loss/fitness should be computed in the EA
    
    # --------------------------------------------------------------
    # EA HYPERPARAMS
    # --------------------------------------------------------------
    num_generations: int # number of generations
    population_size: int # population size (number of individuals)
    p_crossover: float # crossover probability
    mutation_rate: float # mutation rate
    k: int # number of individuals selected for tournament
    dimensions: int # size of the weights vector