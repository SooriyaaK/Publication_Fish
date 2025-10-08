import numpy as np
import pytest
from enums.wall_behaviour_enum import WallBehaviourType
from experiments.wall_behaviour import WallBehaviour


@pytest.fixture
def dummy_data_dict():
    """Create a simple data_dict with radius, tangent, wall distance for 1 fish"""
    return {
        1: {
            "radius_vectors": np.array([[1.0, 0.0]]),   # fish_id 0
            "tangent_vectors": np.array([[0.0, 1.0]]),
            "wall_distances": np.array([0.05])
        }
    }


def test_repulsion_only_below_threshold(dummy_data_dict):
    wb = WallBehaviour()
    vel = wb.compute_wall_velocity(WallBehaviourType.REPULSION_ONLY, dummy_data_dict, t=1, fish_id=0)
    # wall distance < 0.035? In fixture 0.05 > 0.035 → expect zero velocity
    assert isinstance(vel, np.ndarray)
    assert np.allclose(vel, [0.0, 0.0])


def test_repulsion_only_above_threshold():
    wb = WallBehaviour()
    data_dict = {
        1: {
            "radius_vectors": np.array([[1.0, 0.0]]),
            "tangent_vectors": np.array([[0.0, 1.0]]),
            "wall_distances": np.array([0.02])  # < 0.035 → tangent_vector returned
        }
    }
    vel = wb.compute_wall_velocity(WallBehaviourType.REPULSION_ONLY, data_dict, t=1, fish_id=0)
    assert np.allclose(vel, [0.0, 1.0])


def test_align_only_below_threshold():
    wb = WallBehaviour()
    data_dict = {
        1: {
            "radius_vectors": np.array([[1.0, 0.0]]),
            "tangent_vectors": np.array([[0.0, 1.0]]),
            "wall_distances": np.array([0.1])  # < align=0.125
        }
    }
    vel = wb.compute_wall_velocity(WallBehaviourType.ALIGN_ONLY, data_dict, t=1, fish_id=0)
    assert isinstance(vel, np.ndarray)
    # velocity should be a mix of radial and tangent vectors
    assert not np.allclose(vel, [0.0, 0.0])


def test_align_only_above_threshold():
    wb = WallBehaviour()
    data_dict = {
        1: {
            "radius_vectors": np.array([[1.0, 0.0]]),
            "tangent_vectors": np.array([[0.0, 1.0]]),
            "wall_distances": np.array([0.2])  # > align → zero
        }
    }
    vel = wb.compute_wall_velocity(WallBehaviourType.ALIGN_ONLY, data_dict, t=1, fish_id=0)
    assert np.allclose(vel, [0.0, 0.0])


def test_repulsion_and_alignment_below_repulsion():
    wb = WallBehaviour()
    data_dict = {
        1: {
            "radius_vectors": np.array([[1.0, 0.0]]),
            "tangent_vectors": np.array([[0.0, 1.0]]),
            "wall_distances": np.array([0.02])  # < repulsion
        }
    }
    vel = wb.compute_wall_velocity(WallBehaviourType.REPULSION_AND_ALIGNMENT, data_dict, t=1, fish_id=0)
    assert np.allclose(vel, [0.0, 1.0])


def test_repulsion_and_alignment_between():
    wb = WallBehaviour()
    data_dict = {
        1: {
            "radius_vectors": np.array([[1.0, 0.0]]),
            "tangent_vectors": np.array([[0.0, 1.0]]),
            "wall_distances": np.array([0.05])  # between 0.035 and 0.125
        }
    }
    vel = wb.compute_wall_velocity(WallBehaviourType.REPULSION_AND_ALIGNMENT, data_dict, t=1, fish_id=0)
    assert isinstance(vel, np.ndarray)
    # should be a combination, not zero
    assert not np.allclose(vel, [0.0, 0.0])


def test_repulsion_and_alignment_above_align():
    wb = WallBehaviour()
    data_dict = {
        1: {
            "radius_vectors": np.array([[1.0, 0.0]]),
            "tangent_vectors": np.array([[0.0, 1.0]]),
            "wall_distances": np.array([0.2])  # > align
        }
    }
    vel = wb.compute_wall_velocity(WallBehaviourType.REPULSION_AND_ALIGNMENT, data_dict, t=1, fish_id=0)
    assert np.allclose(vel, [0.0, 0.0])


def test_default_case():
    wb = WallBehaviour()
    data_dict = {
        1: {
            "radius_vectors": np.array([[1.0, 0.0]]),
            "tangent_vectors": np.array([[0.0, 1.0]]),
            "wall_distances": np.array([0.05])
        }
    }
    vel = wb.compute_wall_velocity(None, data_dict, t=1, fish_id=0)
    assert np.allclose(vel, [0.0, 0.0])
