import csv, string
import math
import numpy as np
from typing import List
import pandas as pd

from data_utils.data_storage import DataStorageHandler
from enums.datasets_enum import Dataset
from experiments.wall_behaviour import WallBehaviour

class DataPreparator:

    def __init__(self, data_set:Dataset, save_path:string):
        self.data_set = data_set
        self.save_path = save_path
        self.storage_handler = DataStorageHandler()

    def prepare_data(self):
        match self.data_set:
            case Dataset.LEI2020:
                self._prepare_data_lei_2020()

    def _prepare_data_lei_2020(self):
        df = self.storage_handler.load_csv_file_to_dataframe(self.data_set.file_path, self.data_set.headers)
        # add velocities
        df = self._compute_and_append_velocities(df)

        # add sorting information
        df = self._compute_and_append_metrics(df)

        # add timesteps
        df = self._add_timesteps_from_kicktimes(df)

        self.storage_handler.save_data_to_csv(df, self.save_path)

    def _compute_and_append_velocities(self, df):
        df["vx"] = (
            df.groupby(["exp_id", "fish_id"])["x"].diff() /
            df.groupby(["exp_id", "fish_id"])["kicktime"].diff()
        )

        df["vy"] = (
            df.groupby(["exp_id", "fish_id"])["y"].diff() /
            df.groupby(["exp_id", "fish_id"])["kicktime"].diff()
        )

        df.fillna(0)
        return df
    
    def _compute_and_append_metrics(self, df):
        # Compute orientation for all fish
        df["orientation"] = np.arctan2(df["vy"], df["vx"])

        for focal_fish_id in df["fish_id"].unique():
            # Step 1: get focal fish positions + orientation
            focal = (
                df[df["fish_id"] == focal_fish_id]
                .rename(columns={"x": "x_focal", "y": "y_focal", "orientation": "orient_focal"})
                [["exp_id", "kicktime", "x_focal", "y_focal", "orient_focal"]]
            )

            # Step 2: merge with all fish on (exp_id, kicktime)
            df_with_focal = df.merge(focal, on=["exp_id", "kicktime"], how="left")

            # Step 3a: Euclidean distance
            df_with_focal[f"dist_to_fish_{focal_fish_id}"] = np.sqrt(
                (df_with_focal["x"] - df_with_focal["x_focal"])**2 +
                (df_with_focal["y"] - df_with_focal["y_focal"])**2
            )

            # Step 3b: Bearing relative to focal
            df_with_focal[f"bearing_to_fish_{focal_fish_id}"] = np.arctan2(
                df_with_focal["y"] - df_with_focal["y_focal"],
                df_with_focal["x"] - df_with_focal["x_focal"]
            )

            # Step 3c: Orientation difference (neighbor - focal, wrapped to [-pi, pi])
            diff = df_with_focal["orientation"] - df_with_focal["orient_focal"]
            df_with_focal[f"orient_diff_to_fish_{focal_fish_id}"] = (diff + np.pi) % (2*np.pi) - np.pi

            # Clean up temp columns
            df_with_focal = df_with_focal.drop(["x_focal", "y_focal", "orient_focal"], axis=1)

            df = df_with_focal

        return df
    
    def _add_timesteps_from_kicktimes(self, df):
        df["timestep"] = (
            df.groupby("exp_id")["kicktime"]
            .rank(method="dense")  # assign 1,2,3,... for unique kicktimes
            .astype(int) - 1       # shift so smallest = 0
        )
        return df



    def calculate_velocity(self, times: List[float], x_coordinate:List[float], y_coordinate:List[float]) -> List[float]:
        '''
        This function calculates the velocity between two time steps.
        '''
        velocities = []

        for i in range(1, len(times)):
            delta_x = x_coordinate[i] - x_coordinate[i - 1]
            delta_y = y_coordinate[i] - y_coordinate[i - 1]
            distance = np.sqrt(delta_x**2 + delta_y**2)

            delta_t = times[i] - times[i - 1]

            if delta_t <= 0:
                velocity = 0
            else:
                velocity = distance / delta_t

            velocities.append(velocity)
            
        velocities.insert(0, 0)

        return velocities   
    
    def velocity_vector(self, times: List[float], x_coordinate: List[float], y_coordinate:List[float]) -> List[float]:
        """
        Compute the velocity vector.
        time: list of kick time in float.
        x_coordinate: list of float values containing x positions.
        y_coordinate: list of float values containing y positions.
        returns: Returns the velocity computed between two points for vx and vy.

        """
        vx = [0.0]
        vy = [0.0]

        for i in range(1, len(times)):
            delta_x = x_coordinate[i] - x_coordinate[i - 1]
            delta_y = y_coordinate[i] - y_coordinate[i - 1]

            vx.append(delta_x)
            vy.append(delta_y)


        return vx, vy   
    
    def calculate_distance(self, x1, y1, x2, y2):
        '''
        The function calculates the euclidean distance between 2 fish.
        x1, y1: Coordinates of fish 1.
        x2, y2: Coordinates of fish 2.
        '''
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def calculate_bearing(self, x1, y1, x2, y2):
        '''
        Calculates the bearing between the fishes.
        x1, y1: Coordinates of fish 1.
        x2, y2: Coordinates of fish 2.
        '''
        angle_rad = math.atan2(y2 - y1, x2 - x1)
        return (angle_rad + 2 * math.pi) % (2 * math.pi)
    
    def calculate_angle(self, vx: float, vy: float) -> float:
        '''
        Calculate the angle based on the (vx, vy) velocity vector, which points in the direction the fish is moving at a given moment from x- achses.
        '''
        return np.arctan2(vy, vx)

    def orientation_difference(self, angle_focal, angle_neighbour):
        '''
        Calculate for each time step, the orientation difference between the focal fish and its neighbours by computing the angle between their velocity vectors.
        '''
        difference = angle_focal - angle_neighbour
        while difference > np.pi:
            difference -= 2 * np.pi 

        while difference < - np.pi:
            difference += 2 * np.pi 

        return difference
    
    
    def process_data(self, file_path, focal_fish=3):
        '''
        The function creates the feature.csv file. 
        '''

        with open(file_path) as csv_file:
            reader = csv.DictReader(csv_file, delimiter=';')
            data = []
            for row in reader:
                data.append(row)
            # print(f"Total rows read: {len(data)}")

        # Step 1: Group by Experiment_ID and Agent_ID
        features: List[any] = []
        experiment_agent = {}

        for row in data:
            exp_id = int(row['Experiment_ID'])
            agent_id = int(row['Agent_ID'])
            key = (exp_id, agent_id)
            if key not in experiment_agent:
                experiment_agent[key] = []
            experiment_agent[key].append(row)

        
        # Step 2: Sort rows by Kick_Time to calculate velocities and angles

        for key in experiment_agent:
            rows = experiment_agent[key]
            
            rows.sort(key=lambda r: float(r['Kick_Time']))
            kick_times = []
            x_coordinate = []
            y_coordinate = [] 

            for row in rows:
                kick_times.append(float(row['Kick_Time']))
                x_coordinate.append(float(row['X']))
                y_coordinate.append(float(row['Y']))

            velocities = self.calculate_velocity(kick_times, x_coordinate, y_coordinate)
            vx, vy = self.velocity_vector(kick_times, x_coordinate, y_coordinate)

            for i in range(len(rows)):
                velocity = velocities[i]
                angle = self.calculate_angle(vx[i], vy[i])
                
                rows[i]['Velocity_Between_Kicks'] = velocity
                rows[i]['Angle'] = angle
                rows[i]['vx'] = vx[i]
                rows[i]['vy'] = vy[i]

        # Step 3 Group by Experiment ID & Kick Time 
        exp_time = {}

        for row in data:
            exp_id = int(row['Experiment_ID'])
            kick_time = float(row['Kick_Time'])
            key = (exp_id, kick_time)
            if key not in exp_time:
                exp_time[key] = []
            exp_time[key].append(row)
        
        # Step 4 Compare all agents to focal fish 

        for key in exp_time:
            group = exp_time[key]

            focal_row = None 
            for row in group:
                if int(row['Agent_ID']) == focal_fish:
                    focal_row = row 
                    break 

            if focal_row is None:
                continue 

            focal_x_coordinate = float(focal_row['X'])
            focal_y_coordinate = float(focal_row['Y'])
            focal_angle = float(focal_row['Angle'])
            focal_vx = float(focal_row['vx']) 
            focal_vy = float(focal_row['vy']) 

            for row in group:

                if int(row['Agent_ID']) == focal_fish:
                    focal_copy = row.copy()
                    focal_copy['Distance_To_Focal'] = 0.0
                    focal_copy['Bearing_To_Focal'] = 0.0
                    focal_copy['Orientation_Difference_To_Focal'] = 0.0
            
                    features.append(focal_copy)

                    
                
                else:
                    neighbour_copy = row.copy()
                    x_coordinate = float(row['X'])
                    y_coordinate = float(row['Y'])
                    angle = float(row['Angle'])

                    neighbour_copy['Distance_To_Focal'] = self.calculate_distance(focal_x_coordinate, focal_y_coordinate, x_coordinate, y_coordinate)
                    neighbour_copy['Bearing_To_Focal'] = self.calculate_bearing(focal_x_coordinate, focal_y_coordinate, x_coordinate, y_coordinate)
                    neighbour_copy['Orientation_Difference_To_Focal'] = self.orientation_difference(focal_angle, angle)

                    
                    neighbour_copy['vx'] = float(row['vx'])
                    neighbour_copy['vy'] = float(row['vy'])
                    features.append(neighbour_copy)

        fieldnames = []

        for row in features:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)

        with open("features.csv", mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()
            writer.writerows(features)

    def add_step_count(self, file_path):
        '''
        Adds a column step to the feature.csv file.
        
        '''
        with open(file_path, mode='r', newline='') as f:
            reader = csv.DictReader(f, delimiter=';')
            data = list(reader)
            
            fieldnames = reader.fieldnames

            if 'Step' not in fieldnames:
                fieldnames += ['Step']
        
        # Sort the data first by Experiment_ID, then Agent_ID, then Kick_Time
        data = sorted(data, key=lambda row: (int(row['Experiment_ID']), int(row['Agent_ID']), float(row['Kick_Time'])))

        prev_experiment = None
        prev_fish = None
        prev_kick_time = None
        step = 0

        for row in data:
            current_experiment = int(row['Experiment_ID'])
            current_fish = int(row['Agent_ID'])
            current_kick_time = float(row['Kick_Time'])

            # Reset step count if Experiment_ID or Agent_ID changes
            if current_experiment != prev_experiment or current_fish != prev_fish:
                step = 1  # start fresh for each new Experiment+Agent combination
            else:
                # Increment step if Kick_Time increases
                if current_kick_time > prev_kick_time:
                    step += 1
                else:
                    step = 1  # Reset step if Kick_Time decreases or stays the same

            row['Step'] = step

            # Update previous values for the next iteration
            prev_experiment = current_experiment
            prev_fish = current_fish
            prev_kick_time = current_kick_time

        # Write the updated data back to the file
        with open(file_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()
            writer.writerows(data)

    def orientation_velocity(self, file_path):
        kick_times = {}
        with open(file_path) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter = ",")

            for row in csv_reader:
                # experiment_id = int(row['Experiment_ID'])
                kick_time = float(row["Kick_Time"])
                agent_id = int(row["Agent_ID"])
                vx = float(row["vx"])
                vy = float(row["vy"])

                # if experiment_id not in experiments:
                #     experiments[experiment_id] = {}
                
                if kick_time not in kick_times:
                    kick_times[kick_time] = []
                kick_times[kick_time].append((agent_id, vx, vy))
        return kick_times
    
    def compute_csv(self, file_path, agent_id, experiment_id):
        """
        The following function generates a file for experiement 151, specifically for fish id 3,  which contains all necessary information used for the objective function.
        agent_id: The ID of the focal fish. Agent ID = 3.
        experiment_id: The filtered Experiment ID 151.
        """
        
        data_dict = {}

        with open('features.csv') as f:
            csv_reader = csv.DictReader(f)

            for row in csv_reader:
                if int(row['Agent_ID']) == agent_id and int(row['Experiment_ID']) == experiment_id: 
                    vx = float(row['vx'])
                    vy = float(row['vy'])
                    x = float(row['X'])
                    y = float(row['Y'])
                    kicktime = float(row['Kick_Time'])

                    wall_behaviour = WallBehaviour()

                    radius_vector, tangent, x_wall, y_wall, wall_distance = wall_behaviour.distance_wall(x, y, vx, vy, kicktime)

                    if kicktime not in data_dict:
                        data_dict[kicktime] = {
                            'x_coords': [],
                            'y_coords': [],
                            'vx_list': [],
                            'vy_list': [],
                            'wall_x' : [],
                            'wall_y' : [], 
                            'tangent' : [], 
                            'radius_vector' : [],
                            'wall_distance' : []
                        
                        }

                    data_dict[kicktime]['x_coords'].append(x)
                    data_dict[kicktime]['y_coords'].append(y)

                    data_dict[kicktime]['vx_list'].append(vx)
                    data_dict[kicktime]['vy_list'].append(vy)
                    data_dict[kicktime]['wall_x'].append(x_wall)
                    data_dict[kicktime]['wall_y'].append(y_wall)
                    data_dict[kicktime]['tangent'].append(tangent)
                    data_dict[kicktime]['radius_vector'].append(radius_vector)
                    data_dict[kicktime]['wall_distance'].append(wall_distance)

        fieldnames = ['Kick_Time', 'x', 'y', 'vx', 'vy', 'vector_x', 'vector_y', 'x_wall', 'y_wall','tangent_x', 'tangent_y', 'radius_x', 'radius_y', 'wall_distance']
        rows_to_write = []
        for kicktime, data in data_dict.items():
            length = len(data['x_coords'])
            for i in range(length):
                rows_to_write.append({
                    'Kick_Time': kicktime,
                    'x': data['x_coords'][i],
                    'y': data['y_coords'][i],
                    'vx': data['vx_list'][i],
                    'vy': data['vy_list'][i],
                    'x_wall': data['wall_x'][i],
                    'y_wall': data['wall_y'][i],
                    'tangent_x': data['tangent'][i][0],
                    'tangent_y': data['tangent'][i][1],
                    'radius_x': data['radius_vector'][i][0],
                    'radius_y': data['radius_vector'][i][1],
                    'wall_distance': data['wall_distance'][i]
                })

        import os

        csv_output_path = f'features_{experiment_id}.csv'

        header_exists = False
        if os.path.exists(csv_output_path):
            with open(csv_output_path, 'r') as csvfile:
                first_line = csvfile.readline().strip()
                header_exists = first_line == ",".join(fieldnames)


        with open(csv_output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not header_exists:
                writer.writeheader()
            writer.writerows(rows_to_write)

        return data_dict
