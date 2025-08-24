from enum import Enum

"""
Enum containing different reaction behaviours to the presence of a wall
"""
class WallBehaviourType(str, Enum):
    NO_WALL = "no_wall",
    REPULSION_ONLY = "align_only",
    REPULSION_AND_ALIGNMENT = "rep_align"