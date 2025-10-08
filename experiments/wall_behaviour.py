import matplotlib.pyplot as plt
import numpy as np
import random
import csv
import math
from typing import Dict


from enums.wall_behaviour_enum import WallBehaviourType

class WallBehaviour:

    def compute_wall_velocity(self, wall_behaviour_type:WallBehaviourType, data_dict:Dict, t:int, exp_id:int, fish_id:int):
        radial_vector = np.array(data_dict[exp_id][t]['radius_vectors'][fish_id])
        tangent_vector = np.array(data_dict[exp_id][t]['tangent_vectors'][fish_id])
        wall_distance = data_dict[exp_id][t]['wall_distances'][fish_id]
        match wall_behaviour_type:
            case WallBehaviourType.REPULSION_ONLY:
                velocity = self._compute_velocity_for_repulsion_zone(wall_distance=wall_distance,
                                                                     tangent_vector=tangent_vector)
            case WallBehaviourType.ALIGN_ONLY:
                velocity = self._compute_velocity_for_alignment_zone(wall_distance=wall_distance,
                                                                     radial_vector=radial_vector,
                                                                     tangent_vector=tangent_vector)
            case WallBehaviourType.REPULSION_AND_ALIGNMENT:
                velocity = self._compute_velocity_for_repulsion_and_alignment_zone(wall_distance=wall_distance,
                                                                                   radial_vector=radial_vector,
                                                                                   tangent_vector=tangent_vector)
            case _:
                velocity = [0.0, 0.0]
        return np.array(velocity)

    def _compute_velocity_for_repulsion_zone(self, wall_distance, tangent_vector):
        repulsion = 0.035
        if wall_distance < repulsion:
            return tangent_vector
        else:
            return np.array([0.0, 0.0])
        
    def _compute_velocity_for_alignment_zone(self, wall_distance, radial_vector, tangent_vector):
        align = 0.125
        repulsion = 0
        if wall_distance < align:
            smooth = ((repulsion - wall_distance) / (align - repulsion)) * -1
            return (1 - smooth) * radial_vector + smooth * tangent_vector
        else:
            return np.array([0.0, 0.0])
        
    def _compute_velocity_for_repulsion_and_alignment_zone(self, wall_distance, radial_vector, tangent_vector):
        repulsion = 0.035
        align = 0.125
        if wall_distance < repulsion:
            return tangent_vector
    
        elif wall_distance < align:
            smooth = ((repulsion - wall_distance) / (align - repulsion)) * -1
            return (1 - smooth) * radial_vector + smooth * tangent_vector
        else:
            return np.array([0.0, 0.0])
