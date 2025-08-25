import numpy as np
import pytest
from enums.sorting_criteria_enum import SortingCriteria
from enums.loss_functions_enum import LossFunctions
from enums.wall_behaviour_enum import WallBehaviourType

# ⚠️ Adjust this import to match your project
from experiments.objective_function import ObjectiveFunctionEvaluator  # or objective_function if it's a free function


@pytest.mark.parametrize("criterion", [
    SortingCriteria.DISTANCE,
    SortingCriteria.BEARING,
    SortingCriteria.ORIENTATION
])
def test_get_sorted_neighbors(dummy_config, small_data_dict, criterion):
    obj = ObjectiveFunctionEvaluator(dummy_config, small_data_dict)
    # call the function for timestep 1, focal fish 0
    exp_data = small_data_dict[101]
    sorted_idx = obj.get_sorted_neighbors(exp_data, timestep=1, focal_fish_id=0, sorting_criteria=criterion)
    assert isinstance(sorted_idx, np.ndarray)
    assert set(sorted_idx) == {0,1,2}  # indices for 3 fish


def test_cosine_loss(dummy_config, small_data_dict):
    obj = ObjectiveFunctionEvaluator(dummy_config, small_data_dict)
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 0.0])
    assert obj.cosine_loss(a, b) == 0.0

    b = np.array([0.0, 1.0])
    assert obj.cosine_loss(a, b) == 1.0


def test_mse_kicktime(dummy_config, small_data_dict):
    obj = ObjectiveFunctionEvaluator(dummy_config, small_data_dict)
    current = np.array([0.0, 0.0])
    next_pos = np.array([1.0, 1.0])
    pred_vel = np.array([1.0, 0.5])
    mse = obj.mse_kicktime(current, next_pos, pred_vel)
    # mean of [0^2, 0.5^2] = 0.125
    assert np.isclose(mse, 0.125)


@pytest.mark.parametrize("criterion", [
    SortingCriteria.DISTANCE,
    SortingCriteria.BEARING,
    SortingCriteria.ORIENTATION
])
@pytest.mark.parametrize("loss_fn", [
    LossFunctions.COSINE,
    LossFunctions.MSE_KICKTIME
])
def test_objective_function_runs(dummy_config, small_data_dict, criterion, loss_fn):
    obj = ObjectiveFunctionEvaluator(dummy_config, small_data_dict)
    obj.wall_behaviour = None  # or a dummy wall class if testing wall velocity
    obj.config.sorting_criteria = criterion
    obj.config.loss_function = loss_fn

    weights = np.ones(obj.config.dimensions)
    value = obj.objective_function(weights)
    assert isinstance(value, float)
    # Since dataset is small, value should be non-negative
    assert value >= 0.0
