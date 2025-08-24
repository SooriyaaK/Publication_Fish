import matplotlib.pyplot as plt
import numpy as np
import random
import csv
import math


class DataStorageHandler:
    def __init__(self):
        pass

    def load_processed_data(file_path):
        '''
        The function reads the data from the create feature_151.csv file.
        '''

        data_dict = {}

        with open(file_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                kicktime = float(row['Kick_Time'])
                x = float(row['x'])
                y = float(row['y'])
                vx = float(row['vx'])
                vy = float(row['vy'])
                x_wall = float(row['x_wall'])
                y_wall = float(row['y_wall'])
                tangent_x = float(row['tangent_x'])
                tangent_y = float(row['tangent_y'])
                radius_x = float(row['radius_x'])
                radius_y = float(row['radius_y'])
                wall_distance = float(row['wall_distance'])

                if kicktime not in data_dict:
                    data_dict[kicktime] = {
                        'x_coords': [],
                        'y_coords': [],
                        'vx_list': [],
                        'vy_list': [],
                        'wall_vectors': [],
                        'wall_x': [],
                        'wall_y': [],
                        'tangent_vectors': [],
                        'radius_vectors': [],
                        'wall_distances': []
                    }

                data_dict[kicktime]['x_coords'].append(x)
                data_dict[kicktime]['y_coords'].append(y)
                data_dict[kicktime]['vx_list'].append(vx)
                data_dict[kicktime]['vy_list'].append(vy)
                data_dict[kicktime]['wall_x'].append(x_wall)
                data_dict[kicktime]['wall_y'].append(y_wall)
                data_dict[kicktime]['tangent_vectors'].append((tangent_x, tangent_y))
                data_dict[kicktime]['radius_vectors'].append((radius_x, radius_y))
                data_dict[kicktime]['wall_distances'].append(wall_distance)

        return data_dict
    
    def load_data(filepath, data_dict, prediction, focal_agent = 3, experiment_id = 151):
        """
        The function loads the csv data of agent positions, velocities, adds predicted velocities.
        filepath: str. Path to the file.
        data_dict: Dict. Maps actual positions for each kick time.
        prediction: Dict. Predicted velocity vectors for the focal agent at kick time.
        focal_agent: int. ID of the focal fish= 3. 
        experiment_id = 151. int. Experinment ID of the current data.
        return: plot_data : list of dict.
        """
        kicktimes = {}

        # Read the file.
        with open(filepath, newline='') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                if int(row['Experiment_ID']) != experiment_id:
                    continue
                # Extract the data
                kicktime = float(row['Kick_Time'])
                agent_id = int(row['Agent_ID'])
                x = float(row['X'])
                y = float(row['Y'])
                vx = float(row['vx'])
                vy = float(row['vy'])

                if kicktime not in kicktimes:
                    kicktimes[kicktime] = []
                kicktimes[kicktime].append((agent_id, x, y, vx, vy))

        plot_data = []

        sorted_times = sorted(kicktimes.keys())

        for i in range(len(sorted_times) - 1):
            kicktime = sorted_times[i]

            for k in kicktimes[kicktime]:

                if k[0] == focal_agent:
                    plot_data.append({
                        "kicktime": kicktime,
                        "x": k[1],
                        "y": k[2],
                        "actual_vx":k[3],
                        "actual_vy": k[4],
                        "predicted_vx": prediction[i][0],
                        "predicted_vy" :  prediction[i][1]
                    })
        
        return plot_data

    
    def orientation_velocity(file_path):
        kick_times = {}
        with open(file_path) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter = ",")

            for row in csv_reader:
                kick_time = float(row["Kick_Time"])
                agent_id = int(row["Agent_ID"])
                vx = float(row["vx"])
                vy = float(row["vy"])

                if kick_time not in kick_times:
                    kick_times[kick_time] = []
                kick_times[kick_time].append((agent_id, vx, vy))

        return kick_times
    
    def save_to_csv(all_x_values, all_fitness_values, filename="all_runs_best_solution.csv"):
        """Save all x values and their corresponding fitness values from all runs to a single CSV file."""
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write header
            writer.writerow(["Run", "Generation", "Best X Value", "Best Fitness Value"])
            
            # Write the best solution for each generation across all runs
            for run, (x_values, f_values) in enumerate(zip(all_x_values, all_fitness_values), start=1):
                for generation, (best_x, best_fitness) in enumerate(zip(x_values, f_values)):
                    writer.writerow([run, generation, best_x, best_fitness])

    def save_loss_evaluation(evaluation_file):
        '''
        The function evaluate and save loss metrics for each model configurations.
        The function iterates over velocity types, bias configurations, and loss functions.
        It computes the loss values for each combination, and writes the results
        to a CSV file.

        evaluation_file : str. Path to the output CSV file.
        '''
        # loss functions and velocity sorting criteria
        loss_function = ['cosine', 'mse_kicktime', 'mse_trajectory']
        sorted_velocity_types = ['orientation', 'distance', 'bearing']

        # Write the file
        with open(evaluation_file, 'w', newline = '') as f:
            writer = csv.writer(f)
            header = ['Model', 'Fitness EA', 'Weighs', 'Bias_Mode', 'Cosine', 'MSE Kicktime', 'MSE_Trajectory', 'Sorted Velocity Criteria']
            writer.writerow(header) 
            data_dict = load_processed_data("features_151.csv")

            bias_configurations = [
            {
                "name": "bias_wall_zone", 
                "bias_function": bias_wall_zone, 
                "flag": True, 
                "loss": loss_function, 
                "modes": [
                            'bias_wall_zone_repulsion_zone',
                            'bias_wall_zone_alignment_zone',
                            'bias_wall_zone_alignment_domain',
                            'bias_wall_zone_repulsion_alignment_zone',
                            'bias_wall_zone_repulsion_alignment_domain',
                                    ]
            },
            {"name": "bias_zero", "bias_function": bias_zero, "flag": False, "loss": loss_function, "modes": [None]},
            {"name": "bias_random", "bias_function": bias_random, "flag": True, "loss": loss_function, "modes": [None]},
            {"name": "bias_wall", "bias_function": bias_wall, "flag": True, "loss": loss_function, "modes": [None]},
            {"name": "bias_positive", "bias_function": bias_positive, "flag": True, "loss": loss_function, "modes": [None]},
            {"name": "bias_negative", "bias_function": bias_negative, "flag": True, "loss": loss_function, "modes": [None]},
            ]

            for types in sorted_velocity_types:
                # Load the corresponding file
                if types == 'orientation':
                    velocity_file = velocity("orientation_difference_151.csv")
                elif types == 'distance':
                    velocity_file = velocity("distance_difference_151.csv")
                elif types == 'bearing':
                    velocity_file = velocity("bearing_difference_151.csv")
                else:
                    print('File not found')
                    continue
                
                # Test each bias configuration and its modes
                for config in bias_configurations:
                    for mode in config['modes']:
                        # Test each loss function in this configuration
                        for loss in config['loss']:

                            folder_path = r"C:\Users\soori\Desktop\Thesis\Thesis\Test\LossCompare"

                            if mode:
                                file_name = f'{types}_{loss}_results_{mode}'
                            else:
                                file_name = f"{types}_{loss}_results_{config['name']}"
                            
                            # EA results 
                            results = generation_filter(folder_path, file_name, config['flag'], generation = 99)
                            best_result = extract_min_fitness_value(results)
                            
                            # Extract the best weight and its fitness value
                            weights = best_result['weights']
                            fitness = best_result['fitness']
                            model_name = best_result['file_suffix']

                            # Compute losses for all loss functions using objective_function
                            loss_dct = {}
                            for loss in loss_function:

                                loss_value = objective_function(
                                    bias_flag = config['flag'],
                                    bias_func = config['bias_function'],
                                    velocity_file = velocity_file,
                                    data_dict = data_dict,
                                    mode = mode, 
                                    dim = 6,
                                    loss =  loss,
                                    weights = weights,
                                    focal_agent = 3,
                                    experiment_id = 151
                                )

                                loss_dct[loss] = loss_value

                            mode_name = f"{config['name']}_{mode}"

                            writer.writerow([model_name, fitness, weights, mode_name, loss_dct['cosine'], loss_dct['mse_kicktime'], loss_dct['mse_trajectory'], types])
