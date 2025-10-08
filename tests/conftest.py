import os, sys, numpy as np, pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from enums.loss_functions_enum import LossFunctions
from enums.sorting_criteria_enum import SortingCriteria
from enums.wall_behaviour_enum import WallBehaviourType
from dataclasses import dataclass

# dummy wall behaviour
class DummyWall:
    def compute_wall_velocity(self, _):
        return np.array([1.0, 0.0])

@pytest.fixture
def small_data_dict():
    return {
        101: {
            1: {
                "fish_ids": np.array([0,1,2]),
                "vx_list": np.array([1,0.5,-0.5]),
                "vy_list": np.array([0,1,0.5]),
                "x_coords": np.array([0,1,2]),
                "y_coords": np.array([0,1,0]),
                "distances":    np.array([[0,2,1],[2,0,3],[1,3,0]]),
                "bearings":     np.array([[0,0.8,0.2],[0.6,0,0.7],[0.4,0.3,0]]),
                "orient_diffs": np.array([[0,0.6,0.4],[0.6,0,0.5],[0.4,0.5,0]]),
            }
        }
    }

@pytest.fixture
def dummy_config():
    @dataclass
    class Cfg:
        name: str = "test"
        iterations: int = 1
        data_file_path: str = ""
        dataset = None
        experiment_ids = [101]
        fish_ids = [0]
        sorting_criteria = SortingCriteria.DISTANCE
        wall_behaviour = WallBehaviourType.NO_WALL
        loss_function = LossFunctions.COSINE
        num_generations = 1
        population_size = 1
        p_crossover = 0.0
        mutation_rate = 0.0
        k = 1
        dimensions = 6
    return Cfg()
