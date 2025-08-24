import matplotlib.pyplot as plt
from scipy import stats
import csv
import numpy as np
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

class ResultsPlotter:

    def __init__(self):
        pass

    def plot_fitness_over_time(self, all_runs):
        plt.figure(figsize=(10, 6))

        num_generations = len(all_runs[0])  

        for i, run in enumerate(all_runs):
            generations = list(range(num_generations))  
            fitness_values = run  
            plt.plot(generations, fitness_values, label=f'Run {i+1}')
        

        plt.xlabel('Generations')
        plt.ylabel('Best Fitness')
        plt.title('Fitness Over Generations Across All Runs')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_q_q_test(self, cosine):
        plt.figure(figsize=(6, 6))
        stats.probplot(cosine, dist="norm", plot=plt)
        plt.title("Q-Q Plot")
        plt.grid(True, linestyle='--', alpha=0.05)
        plt.tight_layout()
        plt.show()

    def generation_filter(folder_path, file_name, bias_flag = True, generation = 99):
        """
        The function extracts normalized weights and its corresponding fitness values for generation 99 for 20 from a CSV file.

        folder_path : str. Path, where the CSV file is located.
        file_name : str. Base name of the CSV file without '.csv' extension.
        bias_flag : bool. If True, include the bias weight.
        generation : Generation filter in the csv file.
        results : list of tuples.
        """

        filename = f'{file_name}.csv'
        filepath = os.path.join(folder_path, file_name)

        # Read CSV 
        with open(filename, newline='') as f:
            reader = csv.DictReader(f)
            file = []
            for row in reader:
                file.append(row)

        results = []

        # Iterate through all 20 runs and extract the weights and their corresponding fitness value
        for run in range(1, 21):
            for row in file:
                if int(row['Run']) == run and int(row['Generation']) == generation:
                    fitness = float(row['Best Fitness Value'])
                    weights_column = row['Best X Value']
                    
                    weights_str = weights_column.strip('[]')
                    weights_string = weights_str.split()

                    weights = []
                    for weight in weights_string:
                        weights.append(float(weight))

                    # Bias False
                    if not bias_flag:
                        weights = weights[:5]

                    # Normalize the weights L1
                    if np.count_nonzero(weights) > 0:
                        norm = np.linalg.norm(weights, ord = 1)
                        weights = weights / norm

                    if not bias_flag:
                        w1, w2, w3, w4, w5 = weights
                        results.append((file_name, run, fitness, w1, w2, w3, w4, w5))

                    else:
                        w1, w2, w3, w4, w5, w6 = weights

                        results.append((file_name, run, fitness, w1, w2, w3, w4, w5, w6))
            
        return results
    
    def compare_boxplot(cosine_dict, mse_kicktime_dict, mse_trajectory_dict):
        """
        result_dict = {'orientation': [], 'distance': [], 'bearing': []}
        The function creates 3 boxplot for each loss function.
        
        """
        labels = ['Orientation', 'Distance', 'Bearing'] 

        fig, axs = plt.subplots(1, 3, figsize=(10, 10))

        cosine = []
        for label in labels:
            cosine.append(cosine_dict[label.lower()])

        mse_kicktime = []
        for label in labels:
            mse_kicktime.append(mse_kicktime_dict[label.lower()])

        mse_trajectory = []
        for label in labels:
            mse_trajectory.append(mse_trajectory_dict[label.lower()])

        bpc = axs[0].boxplot(cosine, patch_artist = True, showmeans = True) 
        axs[0].set_title('EA Cosine')
        axs[0].set_ylabel('Fitness Value')
        axs[0].set_xticklabels(labels)
        axs[0].grid(True, linestyle='--', alpha=0.5)

        bpk = axs[1].boxplot(mse_kicktime, patch_artist = True, showmeans = True) 
        axs[1].set_title("EA MSE Kicktime")
        axs[1].set_ylabel('Fitness Value')
        axs[1].set_xticklabels(labels)
        axs[1].grid(True, linestyle='--', alpha=0.5)

        bpt =axs[2].boxplot(mse_trajectory, patch_artist = True, showmeans = True) 
        axs[2].set_title('EA MSE Trajectory')
        axs[2].set_ylabel('Fitness Value')
        axs[2].set_xticklabels(labels)
        axs[2].grid(True, linestyle='--', alpha=0.5)

        axs[0].set_xticklabels(labels, rotation=45, ha='right')
        axs[1].set_xticklabels(labels, rotation=45, ha='right')
        axs[2].set_xticklabels(labels, rotation=45, ha='right')
    
        plt.tight_layout()
        plt.savefig('boxplot.svg')
        
        plt.show()

    def plot_weights_per_run(results, file_name):
        """
        Plot weight vectors for each run.
        results: list of tuples (run, fitness, w1, w2, w3, w4, w5, w6)
        file_name : str. name of the file.

        """
        # folder_path = r'C:\Users\soori\Desktop\Thesis\Thesis\Test\LossCompare'
        # file_name = "distance_cosine_results_bias_wall_zone_repulsion_zone"
        # results = generation_filter(folder_path, file_name, True, generation = 99)
        global_min = float('inf')
        global_max = float('-inf')

        for row in results:
            for value in row[3:]:
                if value < global_min:
                    global_min = value
                if value > global_max:
                    global_max = value


        plt.figure(figsize=(10, 10))
        weight_labels = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6']
        x = range(1, 7)  # positions for weights on x-axis

        min_fitness = float('inf')

        # Run with Minimum fitness
        for result in results:
            fitness = result[2]
            if fitness < min_fitness:
                min_fitness = fitness
                best_result = result[3:]
                best_run = result[1]


        for run_data in results:
            run = run_data[1]
            fitness = run_data[2]
            weights = run_data[3:]  # w1 to w6


            if fitness == min_fitness:
                plt.plot(x, weights, marker='o', color = 'black', label=f'Run: {run}')

            else:
                plt.plot(x, weights, marker='o', label=f'Run {run}')
        
        
        plt.xticks(x, weight_labels)
        plt.xlabel('Weights',  fontsize=25)
        plt.ylabel('Weight Values', fontsize=25)
        plt.ylim(global_min - 0.1, global_max + 0.1)
        plt.xlim(0.5, 6.5)
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=12)

        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{file_name}_weight_values_across_runs_at_generation_100.svg")
        plt.savefig(f"{file_name}_weight_values_across_runs_at_generation_100.png")

        plt.savefig(f"{file_name}_weight_values_across_runs_at_generation_100.svg", format="svg", bbox_inches='tight')
        plt.savefig(f"{file_name}_weight_values_across_runs_at_generation_100.png", format="png", dpi=300, bbox_inches='tight')


        plt.show()


    def plot_trajectories(plot_data):
        '''Plot actual vs predicted trajectories for a focal agent - kick times.'''
        actual_x, actual_y = [], []
        pred_x, pred_y = [], []

        x = plot_data[0]['x']
        y = plot_data[0]['y']
        x_pred = x
        y_pred = y

        actual_x.append(x)
        actual_y.append(y)
        pred_x.append(x_pred)
        pred_y.append(y_pred)

        for data in plot_data:
            x += data['actual_vx']
            y += data['actual_vy']
            x_pred += data['predicted_vx']
            y_pred += data['predicted_vy']
            actual_x.append(x)
            actual_y.append(y)
            pred_x.append(x_pred)
            pred_y.append(y_pred)

        fig, ax = plt.subplots(figsize=(10, 10))

        # Fading actual trajectory
        for i in range(1, len(actual_x)):
            alpha = i / len(actual_x)
            ax.plot(actual_x[i-1:i+1], actual_y[i-1:i+1],
                    color='blue', alpha=alpha, linewidth=2, linestyle = ':')

        # Fading predicted trajectory
        for i in range(1, len(pred_x)):
            alpha = i / len(pred_x)
            ax.plot(pred_x[i-1:i+1], pred_y[i-1:i+1],
                    color='red', alpha=alpha, linewidth=2, linestyle = ':')

        # Start and end
        ax.plot(actual_x[0], actual_y[0], 'ko', label='Start')
        ax.plot(actual_x[-1], actual_y[-1], 'b^', label='End Actual')
        ax.plot(pred_x[-1], pred_y[-1], 'r^', label='End Predicted')

        # Tank wall
        circle = Circle((0, 0), 0.25, color='black', fill=False, linewidth=1)
        ax.add_patch(circle)
        ax.set_title('Actual vs Predicted Trajectory')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.axis('equal')
        ax.grid(True, linestyle='--', alpha=0.5)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), loc='upper right')
        plt.tight_layout()
        plt.savefig("trajectory.svg", format="svg", bbox_inches='tight')
        plt.savefig("trajectory.png", format="png", dpi=300, bbox_inches='tight')

    def predicted_velocity(load):
        '''Plot actual vs predicted trajectories for a focal agent - kick times.'''
        kicktimes = []
        x_coords = []
        y_coords = []
        actual_vx = []
        actual_vy = []
        predicted_vx = []
        predicted_vy = []


        for entry in load:
            kicktimes.append(entry['kicktime'])
            x_coords.append(entry['x'])
            y_coords.append(entry['y'])
            actual_vx.append(entry['actual_vx'])
            actual_vy.append(entry['actual_vy'])
            predicted_vx.append(entry['predicted_vx'])
            predicted_vy.append(entry['predicted_vy'])

        plt.figure(figsize=(10,10))
        plt.quiver(x_coords, y_coords, predicted_vx, predicted_vy, color='red', label='Predicted')
        plt.quiver(x_coords, y_coords, actual_vx, actual_vy, color='blue', label='Actual')

        circle_center = (0, 0)
        circle_radius = 0.25

        # Create and add the circle
        circle = Circle(circle_center, circle_radius, color='black', fill=False)
        plt.gca().add_patch(circle)

        plt.gca().set_aspect('equal', adjustable='box')

        # Formatting
        ax.set_aspect('equal')
        plt.xlim(-0.30, 0.30)
        plt.ylim(-0.30, 0.30)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        # ax.set_title(f'Distance')
        ax.grid(True, linestyle='--', alpha=0.5)

        for spine in ax.spines.values():
            spine.set_edgecolor('black')
        ax.tick_params(axis='both', colors='black')  # ticks and tick labels

        plt.tight_layout()


        plt.savefig(f"actual_vs_predicted_velocities_mse_kicktime.png")
        plt.savefig(f"actual_vs_predicted_velocities_mse_kicktime.pdf")
        plt.savefig(f"actual_vs_predicted_velocities_mse_kicktime.svg")
        plt.show()

    def pca(results, file_name):
        """
        The function does PCA on weight vectors across runs and visualize clustering.

        """

        weights = []
        runs = []
        for result in results:
            weights.append(result[3:])
            runs.append(f'R{result[1]}')

        weights_np = np.array(weights)
        
        # Standardize the weights before PCA
        scaled_weights = StandardScaler().fit_transform(weights_np)

        # Reduce dimension to 2 with PCA
        pca_model = PCA(n_components = 2).fit_transform(scaled_weights)

        # Plot PCA
        plt.figure(figsize=(10, 10))
        ax.scatter(pca_model[:, 0], pca_model[:, 1], color='steelblue', s=40)

        for i, label in enumerate(runs):
            plt.text(pca_model[i, 0], pca_model[i, 1], label, fontsize=9)
        
        cluster = AgglomerativeClustering(n_clusters=3, linkage='ward')  
        labels = cluster.fit_predict(pca_model)

        sns.scatterplot(x=pca_model[:, 0], y=pca_model[:, 1], hue=labels, palette='Set2', s=60)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'PCA of Weights - Agglomerative Clustering', fontsize = 25)
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{file_name}_PCA_weights.png")
        plt.savefig(f"{file_name}_PCA_weights.svg")
        plt.savefig(f"{file_name}_PCA_weights.pdf")
        plt.show()

    def plot_pca_with_vectors(self):
        # Draw vectors to understand PCA better

        # 1. Extract weights/runs
        weights = []
        runs = []
        for res in results:
            weight_vector = res[3:]
            weights.append(weight_vector)
            runs.append('R{}'.format(res[1]))  

        W = np.vstack(weights)

        # 2. Standardize & PCA
        scaled = StandardScaler().fit_transform(W)
        pca    = PCA(n_components=2)
        scores = pca.fit_transform(scaled)
        loadings = pca.components_.T  # shape (features=6, 2)

        # 3. Clustering
        labels = AgglomerativeClustering(n_clusters=5, linkage='ward').fit_predict(scores)

        # 4. Plot
        plt.figure(figsize=(8,8))
        ax = plt.gca()

        # Scatter with clusters
        sns.scatterplot(x=scores[:,0], y=scores[:,1], hue=labels, palette='Set2', s=60, ax=ax)

        for i, label in enumerate(runs):
            ax.text(scores[i,0], scores[i,1], label, fontsize=9)

        # Draw loading vectors
        feature_names = ['w1','w2','w3','w4','w5','bias']
        for i, (lx, ly) in enumerate(loadings):
            ax.arrow(0, 0, lx, ly, head_width=0.05, length_includes_head=True, color='black')
            ax.text(lx*1.15, ly*1.15, feature_names[i], color='black', fontsize=12)

        # Final touches
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_title('PCA Biplot of Weights with Clustering')
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()