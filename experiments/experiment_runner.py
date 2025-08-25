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

    def run_experiments(self):
        print(f"running experiments with {self.config.name}")
        
        data_dict = self.storage_handler.load_processed_data(file_path=self.config.data_file_path,
                                                             exp_ids=self.config.experiment_ids)
        
        all_runs_f = []
        all_runs_x = []

        for _ in range(self.config.iterations):
            ea = EA(self.config, data_dict)

            x_best, f_best = ea.run()
            
            all_runs_f.append(f_best)
            all_runs_x.append(x_best)
    
        filename = f"{self.save_location}{self.config.name}_results.csv"
        self.storage_handler.save_to_csv(all_runs_x, all_runs_f, filename=filename)
