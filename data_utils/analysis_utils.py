import csv
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import os
from datetime import datetime
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Circle, Wedge
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

def extract_min_fitness_value(self, results):
    '''
    Extract from the csv file the corresponding min fitnessvalue and their respective weights 
    '''    
    min_fitness = float('inf')
    best_result = {}

    for result in results:
        fitness = result[2]
        if fitness < min_fitness:
            min_fitness = fitness
            
            best_result = {
                'file_suffix' : result[0],
                'run' : result[1],
                'fitness' : min_fitness,
                'weights' : result[3:]
            }
    return best_result   

def collect(folder_path):
    """
    Extracts best fitness values grouped by velocity for a given loss type.
    """
    cosine_dict = {'orientation': [], 'distance': [], 'bearing': []}
    mse_trajectory_dict = {'orientation': [], 'distance': [], 'bearing': []}
    mse_kicktime_dict = {'orientation': [], 'distance': [], 'bearing': []}

    velocity_types = ['orientation', 'distance', 'bearing']
    loss_function = ['cosine', 'mse_kicktime', 'mse_trajectory']

    configurations = [
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

    for velocity in velocity_types:
        for config in configurations:
            for mode in config['modes']:
                for loss in loss_function:

                    if mode:
                        file_name = f'{velocity}_{loss}_results_{mode}'
                    else:
                        file_name = f"{velocity}_{loss}_results_{config['name']}"
        
                    results = generation_filter(folder_path, file_name, config['flag'], generation = 99)
                    best_result = extract_min_fitness_value(results)

                    if loss == 'cosine':
                        cosine = best_result.get('fitness')
                        cosine_dict[velocity].append(cosine)
                    elif loss == 'mse_trajectory':
                        mse_trajectory = best_result.get('fitness')
                        mse_trajectory_dict[velocity].append(mse_trajectory)
                    elif loss == 'mse_kicktime':
                        mse_kicktime = best_result.get('fitness')
                        mse_kicktime_dict[velocity].append(mse_kicktime)

                    else:
                        print('Loss not found')
                   
    return cosine_dict, mse_kicktime_dict, mse_trajectory_dict

def predict(bias_flag, bias_func, velocity_file, data_dict, mode, dim, loss, weights, focal_agent = 3, experiment_id = 151):  
    '''
    The function generate predicted velocities and positions for a focal fish per kick time.
    bias_flag : Bool. If True bias wall will be computed.
    bias_func : Function to compute bias vector.
    velocity_file : Dict. Maps each kicktime to lists of (agent_id, vx, vy) tuples.
    data_dict : Dict. Maps Kicktime to 'x_coords' and 'y_coords'.
    mode : str. None if bias wall is not used, else string to configure the bias wall.
    dim : Int. Focal fish and bias vector, input 6 velocity vector.
    loss : Str. Loss type: cosine, mse_kicktime, or mse_trajectory.
    weights : List. np.array of weights - scaled for the velocity.
    focal_agent : Int.ID of the focal fish.
    experiment_id : Int. ID of the experiment.
    returns: predicted_velocity_list : list of np.ndarray. List of Predicted velocity vectors for the focal agent.
    predicted_position_list : list of np.ndarray. List of predicted position vectors for the focal agent.
    '''
    predicted_velocity_list = []
    predicted_position_list = []
    
    # if not bias_flag = not False = True - ignore 6 weight
    if not bias_flag:
        weights = weights[:5]
        print(f'weights 5 base case: {weights}')

    if not bias_flag:
        w1, w2, w3, w4, w5 = weights
        # print(w1,w2,w3,w4,w5)
    else:
        w1, w2, w3, w4, w5, w6 = weights
        # print(w1,w2,w3,w4,w5,w6)
            
    alpha, beta, gamma, delta, epsilon = w1, w2, w3, w4, w5

    if bias_flag:
        bias = w6
    else:
        bias = 0.0

    weights = np.array([alpha, beta, gamma, delta, epsilon, bias])
        
    # Normalize L1
    if np.count_nonzero(weights) > 0:
        norm = np.linalg.norm(weights, ord = 1)
        weights = weights / norm
        
    kicktimes = []
    
    for key in velocity_file:
        kicktimes.append(key)
    
    # Initialize predicted velocity as the focal agent's first velocity from the actual data
    for agent_id, vx, vy in velocity_file[kicktimes[0]]:
        if agent_id == focal_agent:
            predicted_velocity = np.array([vx, vy])
            break

    # initalize position at timestep 0 
    first_kicktime = np.min(kicktimes)
    initial_x = data_dict[first_kicktime]['x_coords'][0]  # focal agent is at position 0
    initial_y = data_dict[first_kicktime]['y_coords'][0]

    predicted_position = np.array([initial_x, initial_y])
    # print(predicted_position)

    for i in range(len(kicktimes)-1):
        kicktime = kicktimes[i]
        next_kicktime = kicktimes[i + 1]
        
        # velocity lists at current and next times
        velocities = velocity_file[kicktime]
        next_velocities = velocity_file[next_kicktime]
           
        updated_velocity = []
        for j, (agent_id, vx, vy) in enumerate(next_velocities):
            if j == 0:
                updated_velocity.append((agent_id, predicted_velocity[0], predicted_velocity[1]))
            else:
                updated_velocity.append((agent_id, vx, vy))

        vx_total = 0.0
        vy_total = 0.0  

        # replace focal fish data with predicted_velocity
        for j, (agent_id, vx, vy) in enumerate(updated_velocity):
            weight = weights[j]
            vx_total += vx * weight  
            vy_total += vy * weight 
        
        # Bias
        if bias_flag:
            bias = bias_func(data_dict, kicktime, predicted_velocity, mode)
            vx_total += bias[0] * weights[5]  # Multiply by the bias agent's weight
            vy_total += bias[1] * weights[5]  # Multiply by the bias agent's weight
        
        # update
        predicted_velocity = np.array([vx_total, vy_total]) 

        # loss
        if loss == 'cosine':
            updated_x = data_dict[kicktimes[i]]['x_coords'][0]  # focal agent is at position 0
            updated_y = data_dict[kicktimes[i]]['y_coords'][0]

            predicted_position = np.array([updated_x, updated_y])
            predicted_position += predicted_velocity
         
        else:

            if loss == 'mse_kicktime':
                updated_x = data_dict[kicktimes[i]]['x_coords'][0]  # focal agent is at position 0
                updated_y = data_dict[kicktimes[i]]['y_coords'][0]

                predicted_position = np.array([updated_x, updated_y])
                predicted_position += predicted_velocity
            
            elif loss == 'mse_trajectory':
                predicted_position += predicted_velocity

        predicted_velocity_list.append(predicted_velocity)
        predicted_position_list.append(predicted_position)

    return predicted_velocity_list

def standardize(data):
    """Return z-score scaled data as numpy array."""
    data = np.array(data).reshape(-1, 1)
    return StandardScaler().fit_transform(data).flatten()