import csv, string
import pandas as pd

class DataStorageHandler:
    # -------------------------------------------------------------------
    # LOADING
    # -------------------------------------------------------------------

    def load_csv_file_to_dataframe(self, file_path, headers):
        return pd.read_csv(file_path, delim_whitespace=True, header=None, names=headers)
    

    def load_processed_data(self, file_path, exp_ids=None, sort_by=None, ascending=True):
        """
        Load processed CSV into a nested dict: {exp_id -> {timestep -> feature arrays}}.
        
        Parameters
        ----------
        file_path : str
            Path to the processed CSV file.
        exp_ids : list[int] or None
            Experiment IDs to include. If None, include all.
        sort_by : str or None
            Column name to sort rows within each kicktime group. If None, no sorting.
        ascending : bool
            Sort order for the sort_by column.
        """
        df = pd.read_csv(file_path)

        # Filter experiments if provided
        if exp_ids is not None:
            df = df[df["exp_id"].isin(exp_ids)]

        data_dict = {}
        for exp_id, exp_group in df.groupby("exp_id"):
            exp_dict = {}
            for timestep, g in exp_group.groupby("timestep"):
                # Sort within each (exp_id, timestep) group if requested
                if sort_by is not None and sort_by in g.columns:
                    g = g.sort_values(by=sort_by, ascending=ascending)

                exp_dict[timestep] = {
                    "fish_ids": g["fish_id"].to_numpy(),   # keep track of fish identity
                    "x_coords": g["x"].to_numpy(),
                    "y_coords": g["y"].to_numpy(),
                    "vx_list": g["vx"].to_numpy(),
                    "vy_list": g["vy"].to_numpy(),
                    "wall_x": g["x_wall"].to_numpy(),
                    "wall_y": g["y_wall"].to_numpy(),
                    "tangent_vectors": g[["tangent_x", "tangent_y"]].to_numpy(),
                    "radius_vectors": g[["repulse_x", "repulse_y"]].to_numpy(),
                    "wall_distances": g["dist_to_wall"].to_numpy(),
                }
            data_dict[exp_id] = exp_dict

        return data_dict
    
    # -----------------------------------------------------------------------
    # SAVING
    # -----------------------------------------------------------------------

    def save_data_to_csv(self, df: pd.DataFrame, file_path:string):
        df.to_csv(file_path)
    
    def save_to_csv(self, all_x_values, all_fitness_values, filename="all_runs_best_solution.csv"):
        """Save all x values and their corresponding fitness values from all runs to a single CSV file."""
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write header
            writer.writerow(["Run", "Generation", "Best X Value", "Best Fitness Value"])
            
            # Write the best solution for each generation across all runs
            for run, (x_values, f_values) in enumerate(zip(all_x_values, all_fitness_values), start=1):
                for generation, (best_x, best_fitness) in enumerate(zip(x_values, f_values)):
                    writer.writerow([run, generation, best_x, best_fitness])