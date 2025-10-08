import numpy as np
from typing import Dict

from configs.config import Config
from enums.loss_functions_enum import LossFunctions
from enums.sorting_criteria_enum import SortingCriteria
from enums.wall_behaviour_enum import WallBehaviourType
from experiments.wall_behaviour import WallBehaviour

class ObjectiveFunctionEvaluator:

    def __init__(self, config: Config, data_dict: Dict):
        self.config = config
        self.data_dict = data_dict
        self.wall_behaviour: WallBehaviour = WallBehaviour()

    def objective_function(self, weights: np.array):
        weights = self._prepare_weights(self.config, weights)
        mse_loss = 0.0

        for exp_id in self.config.experiment_ids:
            if exp_id not in self.data_dict:
                raise Exception(f"Experiment id {exp_id} not in data.")
            exp_data_dict = self.data_dict[exp_id]

            for fish_id in self.config.fish_ids:
                timesteps = sorted(exp_data_dict.keys())
                if len(timesteps) < 2:
                    continue  # cannot compute velocity with <2 timesteps

                for i in range(1, len(timesteps)-1):
                    t = timesteps[i]
                    t_next = timesteps[i+1]

                    actual_next_velocity = np.array([
                        exp_data_dict[t_next]['vx_list'][fish_id],
                        exp_data_dict[t_next]['vy_list'][fish_id]
                    ])

                    # --- compute predicted velocity ---
                    sorted_indices = self.get_sorted_neighbors(exp_data_dict, t, fish_id, self.config.sorting_criteria)
                    vx = exp_data_dict[t]['vx_list'][sorted_indices]
                    vy = exp_data_dict[t]['vy_list'][sorted_indices]

                    predicted_velocity = np.zeros(2)
                    for rank, idx in enumerate(sorted_indices):
                        predicted_velocity[0] += weights[rank] * vx[rank]
                        predicted_velocity[1] += weights[rank] * vy[rank]

                    # --- add wall velocity if enabled ---
                    if self.config.wall_behaviour != WallBehaviourType.NO_WALL:
                        bias = self.wall_behaviour.compute_wall_velocity(self.config.wall_behaviour)
                        predicted_velocity[0] += bias[0] * weights[-1]
                        predicted_velocity[1] += bias[1] * weights[-1]

                    # --- compute loss ---
                    match self.config.loss_function:
                        case LossFunctions.COSINE:
                            mse_loss += self.cosine_loss(actual_velocity=actual_next_velocity,
                                                        predicted_velocity=predicted_velocity)
                        case LossFunctions.MSE_KICKTIME:
                            actual_position = np.array([
                                exp_data_dict[t]['x_coords'][fish_id],
                                exp_data_dict[t]['y_coords'][fish_id]
                            ])
                            actual_next_position = np.array([
                                exp_data_dict[t_next]['x_coords'][fish_id],
                                exp_data_dict[t_next]['y_coords'][fish_id]
                            ])
                            mse_loss += self.mse_kicktime(current_position=actual_position,
                                                        actual_next_position=actual_next_position,
                                                        predicted_velocity=predicted_velocity)
        return mse_loss

    def cosine_loss(self, actual_velocity, predicted_velocity):
        dotproduct = np.dot(actual_velocity, predicted_velocity)

        magnitude_a = np.linalg.norm(actual_velocity)
        magnitude_b = np.linalg.norm(predicted_velocity)

        cosine_similarity = dotproduct / (magnitude_a * magnitude_b)

        return 1 - np.round(cosine_similarity, 4)

    def mse_kicktime(self, current_position, actual_next_position, predicted_velocity):
        predicted_position = np.array(current_position) + np.array(predicted_velocity)
        actual_position = np.array(actual_next_position)

        squared_errors = (predicted_position - actual_position) ** 2
        mse = np.mean(squared_errors)  # mean across x and y
        return mse

    def _prepare_weights(self, config:Config, weights:list):
        # Initalize weight vector c if not provided
        if weights is None:
            weights = np.random.rand(config.dimensions)

        # weights
        alpha, beta, gamma, delta, epsilon = weights[0], weights[1], weights[2], weights[3], weights[4]
        bias = weights[5]
        weights = np.array([alpha, beta, gamma, delta, epsilon, bias])

        # if bias = False -> dime 5 -> bias_zero
        if config.wall_behaviour == WallBehaviourType.NO_WALL:
            weights[5] = 0.0

        # Normalise the weights using L1 norm
        if np.count_nonzero(weights) > 0:
            norm = np.linalg.norm(weights, ord = 1)
            weights = weights / norm

        return weights
    
    def get_sorted_neighbors(self, exp_data_dict, timestep, focal_fish_id, sorting_criteria):
        """
        Return indices of neighbors sorted by the given metric relative to focal fish.

        Parameters
        ----------
        exp_data_dict : dict
            Data for a single experiment.
        timestep : int
            Current timestep.
        focal_fish_id : int
            ID of the focal fish.
        metric : str
            Sorting metric: "distance", "bearing", or "orientation".
        """
        g = exp_data_dict[timestep]
        fish_ids = g["fish_ids"]
        focal_index = np.where(fish_ids == focal_fish_id)[0][0]

        match sorting_criteria:
            case SortingCriteria.DISTANCE:
                values = g["distances"][focal_index]
            case SortingCriteria.ORIENTATION:
                values = g["orient_diffs"][focal_index]
            case SortingCriteria.BEARING:
                values = g["bearings"][focal_index]

        sorted_indices = np.argsort(values)
        return sorted_indices