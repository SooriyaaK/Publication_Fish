import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from data_utils.data_preparation import DataPreparator
from enums.datasets_enum import Dataset

@pytest.fixture
def dummy_df():
    # Minimal DataFrame with 2 experiments, 2 fish, 3 timesteps
    data = {
        "exp_id": [1,1,1,1,1,1,2,2,2,2,2,2],
        "fish_id": [0,0,0,1,1,1,0,0,0,1,1,1],
        "kicktime": [0,1,2,0,1,2,0,1,2,0,1,2],
        "x": [0,1,2,0,0.5,1,0,1,2,0,0.5,1],
        "y": [0,0,0,0,0.5,1,0,0,0,0,0.5,1],
    }
    return pd.DataFrame(data)

@pytest.fixture
def preparator(monkeypatch):
    prep = DataPreparator(data_set=Dataset.LEI2020, save_path="dummy.csv")
    # Mock the storage handler to avoid file I/O
    prep.storage_handler = MagicMock()
    return prep

def test_prepare_data_calls_internal_methods(preparator, dummy_df):
    # Mock internal methods
    preparator._prepare_data_lei_2020 = MagicMock()
    preparator.prepare_data()
    preparator._prepare_data_lei_2020.assert_called_once()

def test_compute_and_append_velocities(preparator, dummy_df):
    df = preparator._compute_and_append_velocities(dummy_df.copy())
    assert "vx" in df.columns
    assert "vy" in df.columns
    # Check that first row has 0 velocity due to diff
    assert df.iloc[0]["vx"] == 0 or np.isnan(df.iloc[0]["vx"])
    assert df.iloc[0]["vy"] == 0 or np.isnan(df.iloc[0]["vy"])

def test_compute_and_append_sorting_metrics(preparator, dummy_df):
    df = preparator._compute_and_append_velocities(dummy_df.copy())
    df = preparator._compute_and_append_sorting_metrics(df)
    # Check that new columns exist for each fish
    for focal in df["fish_id"].unique():
        assert f"dist_to_fish_{focal}" in df.columns
        assert f"bearing_to_fish_{focal}" in df.columns
        assert f"orient_diff_to_fish_{focal}" in df.columns

def test_compute_wall_metrics(preparator, dummy_df):
    df = preparator._compute_wall_metrics(dummy_df.copy())
    for col in ["x_wall", "y_wall", "repulse_x", "repulse_y", "tangent_x", "tangent_y", "dist_to_wall"]:
        assert col in df.columns

def test_add_timesteps_from_kicktimes(preparator, dummy_df):
    df = preparator._add_timesteps_from_kicktimes(dummy_df.copy())
    assert "timestep" in df.columns
    # The first timestep for each experiment should be 0
    for exp in df["exp_id"].unique():
        first_ts = df[df["exp_id"]==exp]["timestep"].min()
        assert first_ts == 0
