from enum import Enum

"""
Enum of sorting criteria for the neighbouring fish
"""
class SortingCriteria(str, Enum):
    DISTANCE = "distance",
    ORIENTATION = "orientation",
    BEARING = "bearing"