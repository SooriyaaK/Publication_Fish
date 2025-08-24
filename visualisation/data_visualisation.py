import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.patches import Circle
import pandas as pd

class DataVisualisator:

    def visualise_entire_experiment(self):
        # Load data
        df = pd.read_csv('features.csv')

        # Filter for Experiment 151
        df_exp = df[df['Experiment_ID'] == 151].copy()

        # Define colors for 5 agents (Agent 3 in blue)
        agent_colors = {
            1: 'orange',
            2: 'green',
            3: 'blue',
            4: 'purple',
            5: 'cyan'
        }

        arrow_len = 0.04
        fig, ax = plt.subplots(figsize=(10, 10))

        # Draw circular tank boundary (radius 0.25)
        tank = Circle((0, 0), 0.25, edgecolor='black', facecolor='none', linewidth=1)
        ax.add_patch(tank)

        # Plot each agent's normalized heading arrows
        for aid, color in agent_colors.items():
            df_agent = df_exp[df_exp['Agent_ID'] == aid]
            x = df_agent['X'].values
            y = df_agent['Y'].values
            vx = df_agent['vx'].values
            vy = df_agent['vy'].values
            
            # Normalize velocity vectors - unit length
            norms = np.sqrt(vx**2 + vy**2)
            norms[norms == 0] = 1
            dx = (vx / norms) * arrow_len
            dy = (vy / norms) * arrow_len
            
            ax.quiver(
                x, y, dx, dy,
                angles='xy', scale_units='xy', scale=1,
                color=color, width=0.003, headwidth=3, headlength=5, alpha=1,linewidths=2,
                label=f'Fish ID {aid}'
            )

        ax.set_aspect('equal')
        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.3, 0.3)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, linestyle='--', alpha=0.5)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), loc='upper right')

        for spine in ax.spines.values():
            spine.set_edgecolor('black')

        ax.tick_params(axis='both', colors='black')
        plt.tight_layout()
        plt.savefig("actual_data.svg", format="svg", bbox_inches='tight')
        plt.savefig("actual_data.png", format="png", dpi=300, bbox_inches='tight')


        plt.show()

    def visualise_partial_experiment(self):
        # Load data
        df = pd.read_csv('features.csv')

        # Filter for Experiment 151
        df_exp = df[df['Experiment_ID'] == 151].copy()

        # Arrow size
        arrow_len = 0.04

        # Randomly select 5 unique steps between 1 and 200
        unique_kicks = df_exp['Kick_Time'].unique().tolist()
        random_kicks = sorted(random.sample(unique_kicks, 5))

        # Assign colors to each kick time
        cmap = plt.get_cmap('tab10')
        step_colors = dict(zip(random_kicks, cmap.colors))

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.add_patch(Circle((0, 0), 0.25, edgecolor='black', facecolor='none', linewidth=1))

        # Plot each kick time
        for step_value, color in step_colors.items():
            df_step = df_exp[df_exp['Kick_Time'] == step_value]

            for _, row in df_step.iterrows():
                x, y, vx, vy = row['X'], row['Y'], row['vx'], row['vy']
                norm = np.sqrt(vx**2 + vy**2) or 1
                dx = (vx / norm) * arrow_len
                dy = (vy / norm) * arrow_len

                ax.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1,
                        color=color, width=0.003, headwidth=3, headlength=5)


        ax.set_aspect('equal')
        ax.set_xlim(-0.30, 0.30)
        ax.set_ylim(-0.30, 0.30)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, linestyle='--', alpha=0.5)

        handles = []
        labels = []

        for step, color in step_colors.items():
            handle = plt.Line2D([0], [0], color=color, lw=3)
            handles.append(handle)
            labels.append(step)

        ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), loc='upper right', title = 'Kick Time')
        plt.tight_layout()

        plt.savefig("data.svg", format="svg", bbox_inches='tight')
        plt.savefig("data.png", format="png", dpi=300, bbox_inches='tight')


    def plot_neighbors_and_focal(self, load, features_csv, experiment_id, target_kicktime, focal_id=3):

        df = pd.read_csv(features_csv)
        df = df[df['Experiment_ID'] == experiment_id]
        df_step = df[df['Kick_Time'] == target_kicktime].copy()

        # 2) focal position
        focal = df_step[df_step['Agent_ID'] == focal_id].iloc[0]
        fx, fy = focal['X'], focal['Y']

        # 3) Compute neighbor distances & rank
        df_step['dist'] = np.hypot(df_step['X'] - fx, df_step['Y'] - fy)
        neigh = df_step[df_step['Agent_ID'] != focal_id].sort_values('dist').copy()
        # assign rank without comprehension
        ranks = []
        for i in range(len(neigh)):
            ranks.append(i + 1)
        neigh['rank'] = ranks

        # 4) Extract predicted & actual focal velocity
        v_act = None
        v_pred = None
        for rec in load:
            if rec.get('kicktime') == target_kicktime:
                v_act = (rec['actual_vx'], rec['actual_vy'])
                v_pred = (rec['predicted_vx'], rec['predicted_vy'])
                break

        arrow_len = 0.04
        rank_colors = {1: 'purple', 2: 'orange', 3: 'green', 4: 'cyan'}

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.add_patch(Circle((0, 0), 0.25, edgecolor='k', fill=False))

        # 6) Plot neighbor vectors
        for idx in range(len(neigh)):
            row = neigh.iloc[idx]
            x = row['X']; y = row['Y']
            vx = row['vx']; vy = row['vy']
            rank = row['rank']
            norm = np.hypot(vx, vy)
            if norm == 0:
                norm = 1.0
            dx = vx / norm * arrow_len
            dy = vy / norm * arrow_len
            color = rank_colors.get(rank, 'gray')
            ax.quiver(x, y, dx, dy, color=color, width=0.003)

        # 7) Plot focal actual vs predicted
        norm_act = np.hypot(v_act[0], v_act[1])
        if norm_act == 0:
            norm_act = 1.0
        norm_pre = np.hypot(v_pred[0], v_pred[1])
        if norm_pre == 0:
            norm_pre = 1.0
        dax = v_act[0] / norm_act * arrow_len
        day = v_act[1] / norm_act * arrow_len
        px = v_pred[0] / norm_pre * arrow_len
        py = v_pred[1] / norm_pre * arrow_len

        ax.quiver(fx, fy, dax, day, color='blue', width=0.004, label='Focal Actual')
        ax.quiver(fx, fy, px,  py,  color='red',  width=0.004, label='Focal Predicted')

        neigh_handles = []
        neigh_labels = []
        for rank in [1, 2, 3, 4]:
            neigh_handles.append(plt.Line2D([0], [0], color=rank_colors[rank], lw=3))
            neigh_labels.append(f"{rank}ᵗʰ Nearest")

        focal_handles = []
        focal_labels = []
        for color, label in [('blue', 'Focal Actual'), ('red', 'Focal Predicted')]:
            focal_handles.append(plt.Line2D([0], [0], color=color, lw=3))
            focal_labels.append(label)

        ax.legend(
            neigh_handles + focal_handles,
            neigh_labels + focal_labels,
            title="Distance",
            loc='upper right'
        )

        ax.set_aspect('equal')
        ax.set_xlim(0, 0.3)
        ax.set_ylim(0, 0.3)
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        plt.show()

    def plot_wall_zones_with_distance_line(self, tank_radius=0.25):
        fig, ax = plt.subplots(figsize=(6,6))

        # Tank boundary
        tank = Circle((0,0), tank_radius, edgecolor='black', facecolor='none', linewidth=1)
        ax.add_patch(tank)
        
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-tank_radius - 0.1, tank_radius + 0.1)
        ax.set_ylim(-tank_radius - 0.1, tank_radius + 0.1)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)

        plt.savefig('cooridinate.png')
        plt.savefig('cooridinate.pdf')
        plt.savefig('cooridinate.svg')