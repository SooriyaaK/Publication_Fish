import math, string
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any
from enums.sorting_criteria_enum import SortingCriteria
from enums.wall_behaviour_enum import WallBehaviourType

@dataclass
class Config:
    data_file_path: string # where the data has been stored

    sorting_criteria: SortingCriteria # how to sort the neighbours
    wall_behaviour: WallBehaviourType # how the focal fish reacts to the presence of a wall
    