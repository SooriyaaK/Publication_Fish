from configs.config_orientation_nowall_cosine import ConfigOrientationNoWallCosine
from experiments.experiment_runner import ExperimentRunner

runner = ExperimentRunner(ConfigOrientationNoWallCosine(), save_location="results/")
runner.run_experiments()