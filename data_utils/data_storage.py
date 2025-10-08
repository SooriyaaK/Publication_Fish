import csv, string
import pandas as pd

class DataStorageHandler:
    # -------------------------------------------------------------------
    # LOADING
    # -------------------------------------------------------------------

    def load_csv_file_to_dataframe(self, file_path, headers):
        return pd.read_csv(file_path, delim_whitespace=True, header=None, names=headers)
    

    def load_processed_data(self, file_path, exp_ids=None, sort_by=None, ascending=True):
        df = pd.read_csv(file_path)

        # Filter experiments if provided
        if exp_ids is not None:
            df = df[df["exp_id"].isin(exp_ids)]

        data_dict = {}
        for exp_id, exp_group in df.groupby("exp_id"):
            exp_dict = {}
            for timestep, g in exp_group.groupby("timestep"):
                if sort_by is not None and sort_by in g.columns:
                    g = g.sort_values(by=sort_by, ascending=ascending)

                # Dynamically detect columns
                dist_cols   = [c for c in g.columns if c.startswith("dist_to_fish_")]
                bearing_cols = [c for c in g.columns if c.startswith("bearing_to_fish_")]
                orient_cols  = [c for c in g.columns if c.startswith("orient_diff_to_fish_")]

                exp_dict[timestep] = {
                    "fish_ids": g["fish_id"].to_numpy(),
                    "x_coords": g["x"].to_numpy(),
                    "y_coords": g["y"].to_numpy(),
                    "vx_list": g["vx"].to_numpy(),
                    "vy_list": g["vy"].to_numpy(),
                    "wall_x": g["x_wall"].to_numpy(),
                    "wall_y": g["y_wall"].to_numpy(),
                    "tangent_vectors": g[["tangent_x", "tangent_y"]].to_numpy(),
                    "radius_vectors": g[["repulse_x", "repulse_y"]].to_numpy(),
                    "wall_distances": g["dist_to_wall"].to_numpy(),
                    "distances": g[dist_cols].to_numpy(),        # shape (n_fish, n_neighbors)
                    "bearings": g[bearing_cols].to_numpy(),      # same shape
                    "orient_diffs": g[orient_cols].to_numpy(),   # same shape
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