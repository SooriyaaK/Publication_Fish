from configs.config_exp151_fish3_orientation_nowall_cosine import ConfigSpecific
from enums.sorting_criteria_enum import SortingCriteria
from enums.loss_functions_enum import LossFunctions
from enums.wall_behaviour_enum import WallBehaviourType
from experiments.experiment_runner import ExperimentRunner

config = ConfigSpecific()
config.sorting_criteria = SortingCriteria.BEARING
config.loss_function = LossFunctions.COSINE

for wall_behaviour_type in [WallBehaviourType.NO_WALL,
                            WallBehaviourType.REPULSION_ONLY,
                            WallBehaviourType.ALIGN_ONLY,
                            WallBehaviourType.REPULSION_AND_ALIGNMENT]:
    config.wall_behaviour = wall_behaviour_type
    config.name = None
    runner = ExperimentRunner(config, save_location="results/")
    runner.run_experiments()