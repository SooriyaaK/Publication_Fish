import string

from configs.config import Config
from data_utils.data_storage import DataStorageHandler
from experiments.ea import EA

import experiments.objective_function as of

class ExperimentRunner:
    def __init__(self, config: Config, save_location: string):
        self.config = config
        self.save_location = save_location
        self.storage_handler = DataStorageHandler()

        if self.config.name == None or self.config.name == "":
            self.config.name = f"{self.config.dataset.label}_exp={self.config.experiment_ids}_ids={self.config.fish_ids}_{self.config.sorting_criteria.value}_{self.config.wall_behaviour.value}_{self.config.loss_function.value}"

    def run_experiments(self):
        print(f"running experiments with {self.config.name}")
        
        data_dict = self.storage_handler.load_processed_data(file_path=self.config.data_file_path,
                                                             exp_ids=self.config.experiment_ids)
        
        all_runs_f = []
        all_runs_x = []

        for iter in range(self.config.iterations):
            print(f"Running iteration {iter+1}/{self.config.iterations} for {self.config.name}")
            ea = EA(self.config, data_dict)

            x_best, f_best = ea.run()
            
            all_runs_f.append(f_best)
            all_runs_x.append(x_best)
    
        filename = f"{self.save_location}{self.config.name}_results.csv"
        self.storage_handler.save_to_csv(all_runs_x, all_runs_f, filename=filename)
