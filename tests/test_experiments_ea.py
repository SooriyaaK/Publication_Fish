import numpy as np
import pytest
from dataclasses import dataclass

# Adjust these imports to your project structure
from experiments.ea import EA
from experiments.objective_function import ObjectiveFunctionEvaluator


# -------------------------
# Dummy config and objective
# -------------------------
@dataclass
class DummyConfig:
    population_size: int = 4
    dimensions: int = 3
    num_generations: int = 2
    p_crossover: float = 1.0
    mutation_rate: float = 0.1
    k: int = 2
    # Needed by ObjectiveFunctionEvaluator
    experiment_ids: list = (0,)
    fish_ids: list = (0,)
    sorting_criteria = None
    wall_behaviour = None
    loss_function = None


class DummyObjectiveEvaluator:
    def __init__(self, config, data_dict):
        self.config = config
        self.data_dict = data_dict

    def objective_function(self, weights):
        # simple sum of squares as dummy loss
        return np.sum(weights**2)


# -------------------------
# Fixtures
# -------------------------
@pytest.fixture
def dummy_data_dict():
    # minimal data structure
    return {0: {1: {"vx_list": np.array([0.0]), "vy_list": np.array([0.0])}}}


@pytest.fixture
def ea_instance(dummy_data_dict):
    cfg = DummyConfig()
    ea = EA(cfg, dummy_data_dict)
    ea.objective_function_evaluator = DummyObjectiveEvaluator(cfg, dummy_data_dict)
    return ea


# -------------------------
# Tests
# -------------------------
def test_initialization(ea_instance):
    pop = ea_instance.initialization(5, 3)
    assert pop.shape == (5, 3)
    assert np.all(pop >= -1) and np.all(pop <= 1)


def test_evaluation(ea_instance):
    x = np.ones((2, 3))
    fitness = ea_instance.evaluation(x)
    assert fitness.shape == (2,)
    assert np.all(fitness == 3)  # sum of squares of ones


def test_mutation(ea_instance):
    x = np.zeros((2, 3))
    mutated = ea_instance.mutation(x.copy(), mutation_rate=0.1)
    assert mutated.shape == x.shape
    assert np.all(mutated >= -1) and np.all(mutated <= 1)


def test_crossover(ea_instance):
    x = np.array([[1, 2, 3], [4, 5, 6]])
    # with p_crossover=1.0, crossover always occurs
    offspring = ea_instance.crossover(x, p_crossover=1.0)
    assert offspring.shape == x.shape


def test_parent_selection(ea_instance):
    x = np.array([[1,2],[3,4],[5,6]])
    f = np.array([0.5,0.2,0.1])
    x_sel, f_sel = ea_instance.parent_selection(x, f, k=2)
    assert x_sel.shape == x.shape
    assert f_sel.shape == f.shape


def test_survivor_selection(ea_instance):
    x = np.array([[1,2],[3,4]])
    f = np.array([1.0, 0.5])
    x_off = np.array([[0.1,0.2],[0.3,0.4]])
    f_off = np.array([0.2, 0.1])
    x_new, f_new = ea_instance.survivor_selection(x, f, x_off, f_off)
    assert x_new.shape == x.shape
    assert f_new.shape == f.shape
    # best fitness should be first
    assert np.min(f_new) <= np.max(f_new)


def test_full_run(ea_instance):
    x_best, f_best = ea_instance.run()
    assert len(x_best) == ea_instance.config.num_generations
    assert len(f_best) == ea_instance.config.num_generations
    for x in x_best:
        assert len(x) == ea_instance.config.dimensions
