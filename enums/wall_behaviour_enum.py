from enum import Enum

"""
Enum containing different reaction behaviours to the presence of a wall
"""
class WallBehaviourType(str, Enum):
    NO_WALL = "nowall",
    REPULSION_ONLY = "repulsiononly",
    ALIGN_ONLY = "alignonly",
    REPULSION_AND_ALIGNMENT = "repulsionalign"