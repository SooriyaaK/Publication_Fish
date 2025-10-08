from configs.config_exp151_fish3_orientation_nowall_cosine import ConfigSpecific
from experiments.experiment_runner import ExperimentRunner

runner = ExperimentRunner(ConfigSpecific(), save_location="results/")
runner.run_experiments()