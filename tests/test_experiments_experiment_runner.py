import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass

import experiments.experiment_runner as er
from experiments.ea import EA


# -------------------------
# Dummy Config
# -------------------------
@dataclass
class DummyConfig:
    name: str = "test"
    iterations: int = 2
    data_file_path: str = "dummy.csv"
    experiment_ids: list = (0,)
    fish_ids: list = (0,)
    population_size: int = 2
    dimensions: int = 3
    num_generations: int = 1
    p_crossover: float = 0.0
    mutation_rate: float = 0.0
    k: int = 1
    sorting_criteria = None
    wall_behaviour = None
    loss_function = None


# -------------------------
# Fixtures
# -------------------------
@pytest.fixture
def dummy_data_dict():
    # minimal structure for EA
    return {0: {1: {"vx_list": [0.0], "vy_list": [0.0]}}}


@pytest.fixture
def runner_instance(monkeypatch, dummy_data_dict):
    cfg = DummyConfig()
    runner = er.ExperimentRunner(cfg, save_location="/tmp/")

    # Mock storage handler methods
    runner.storage_handler.load_processed_data = MagicMock(return_value=dummy_data_dict)
    runner.storage_handler.save_to_csv = MagicMock()
    
    # Mock EA.run to return dummy values
    def dummy_run(self):
        x_best = [[0.1,0.2,0.3]]
        f_best = [0.5]
        return x_best, f_best
    
    monkeypatch.setattr("experiments.ea.EA.run", dummy_run)
    return runner


# -------------------------
# Tests
# -------------------------
def test_run_experiments_calls_load_and_save(runner_instance):
    runner_instance.run_experiments()

    # Check that load_processed_data was called
    runner_instance.storage_handler.load_processed_data.assert_called_once_with(
        file_path=runner_instance.config.data_file_path,
        exp_ids=runner_instance.config.experiment_ids
    )

    # Check that save_to_csv was called
    runner_instance.storage_handler.save_to_csv.assert_called_once()
    args, kwargs = runner_instance.storage_handler.save_to_csv.call_args

    # Save_to_csv uses keyword argument for filename
    all_runs_x = args[0]
    all_runs_f = args[1]
    filename = kwargs["filename"]

    # Should have as many entries as iterations
    assert len(all_runs_x) == runner_instance.config.iterations
    assert len(all_runs_f) == runner_instance.config.iterations

    # Filename should contain experiment name
    assert runner_instance.config.name in filename
