import pytest
import pandas as pd
import numpy as np
from io import StringIO
from unittest.mock import patch, mock_open

from data_utils.data_storage import DataStorageHandler

@pytest.fixture
def sample_csv():
    csv_data = """exp_id,timestep,fish_id,x,y,vx,vy,x_wall,y_wall,tangent_x,tangent_y,repulse_x,repulse_y,dist_to_wall,dist_to_fish_0,dist_to_fish_1,bearing_to_fish_0,bearing_to_fish_1,orient_diff_to_fish_0,orient_diff_to_fish_1
101,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1
101,0,1,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0
"""
    return StringIO(csv_data)

@pytest.fixture
def handler():
    return DataStorageHandler()

def test_load_processed_data(handler, sample_csv):
    # Patch pandas read_csv to read from our string
    with patch("pandas.read_csv", return_value=pd.read_csv(sample_csv)) as mock_read:
        data_dict = handler.load_processed_data("dummy.csv", exp_ids=[101])
        mock_read.assert_called_once_with("dummy.csv")

        # Check structure
        assert 101 in data_dict
        exp_data = data_dict[101]
        for t, t_data in exp_data.items():
            for key in ["fish_ids","x_coords","y_coords","vx_list","vy_list","wall_x","wall_y",
                        "tangent_vectors","radius_vectors","wall_distances","distances","bearings","orient_diffs"]:
                assert key in t_data
            # Check shapes
            assert t_data["fish_ids"].shape[0] == 2  # 2 fish

def test_save_data_to_csv_calls_to_csv(handler):
    df = pd.DataFrame({"a":[1,2]})
    with patch.object(df, "to_csv") as mock_to_csv:
        handler.save_data_to_csv(df, "dummy.csv")
        mock_to_csv.assert_called_once_with("dummy.csv")

def test_save_to_csv_writes_correct_number_of_rows(handler):
    all_x_values = [
        [np.array([0.1,0.2]), np.array([0.3,0.4])],
        [np.array([0.5,0.6])]
    ]
    all_fitness_values = [
        [0.01, 0.02],
        [0.03]
    ]
    m = mock_open()
    with patch("builtins.open", m):
        handler.save_to_csv(all_x_values, all_fitness_values, filename="dummy.csv")

    handle = m()
    # header + sum of all generations (2+1)
    expected_lines = 1 + 3
    written_lines = [call.args[0] for call in handle.write.call_args_list if call.args]
    assert len(written_lines) >= expected_lines  # at least header + rows written
