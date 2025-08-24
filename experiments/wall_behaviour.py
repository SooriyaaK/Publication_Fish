import matplotlib.pyplot as plt
import numpy as np
import random
import csv
import math

class WallBehaviour:

    def bias_wall(self, data_dict, kicktime, heading, mode):
        '''
        The function simulates the wall influence.
        data_dict: Data from the file.
        mode: bool -> wall_zones
        heading: -> predicted velocity vector

        # x_fish = x 
        # y_fish = y 
        # distance_to_origin = math.sqrt(x_fish**2 + y_fish**2)
        # x_wall = (x_fish  / distance_to_origin) * radius
        # y_wall = (y_fish  / distance_to_origin) * radius
        # vector_x = x_wall - x_fish -> radius vector
        # vector_y = y_wall - y_fish -> radius vector
        radius_vector: Difference between wall and fish.
        '''
        
        return np.array(data_dict[kicktime]['radius_vectors'][0])
    
    def bias_wall_zone(self, data_dict, kicktime, heading, mode):
        '''
        The function simulates different fish movements based on the focal fish distance to the wall.
        data_dict: Data from the file.
        mode: bool -> wall_zones
        heading: -> predicted velocity vector
        Different constellation of the wall_vector.
        Repulsion: Push away from the wall - follow the tangent of the wall before crashing.
        Alignment: Attraction zone towards the wall. 
        '''

        radius = 0.25
        
        x_wall = data_dict[kicktime]['wall_x'][0]
        y_wall = data_dict[kicktime]['wall_y'][0]

        radial_vector = np.array(data_dict[kicktime]['radius_vectors'][0])
        tangent_vector = np.array(data_dict[kicktime]['tangent_vectors'][0])
        wall_distance = data_dict[kicktime]['wall_distances'][0]

        if mode == 'repulsion_zone':
            repulsion = 0.035
            if wall_distance < repulsion:
                return tangent_vector
            else:
                return np.array([0.0, 0.0])
        
        elif mode == 'alignment_zone':
            align = 0.125
            repulsion = 0
            if wall_distance < align:
                smooth = ((repulsion - wall_distance) / (align - repulsion)) * -1
                return (1 - smooth) * radial_vector + smooth * tangent_vector
            else:
                return np.array([0.0, 0.0])

        
        elif mode == 'alignment_domain':
            align = radius
            repulsion = 0
            if wall_distance < align:
                smooth = ((repulsion - wall_distance) / (align - repulsion)) * -1
                return (1 - smooth) * radial_vector + smooth * tangent_vector
        
        elif mode == 'repulsion_alignment_zone':
            repulsion = 0.035
            align = 0.125
            if wall_distance < repulsion:
                return tangent_vector
        
            elif wall_distance < align:
                smooth = ((repulsion - wall_distance) / (align - repulsion)) * -1
                return (1 - smooth) * radial_vector + smooth * tangent_vector
            else:
                return np.array([0.0, 0.0])

            
        elif mode == 'repulsion_alignment_domain':
            repulsion = 0.035
            align = 0.25
            if wall_distance < repulsion:
                return tangent_vector
        
            elif wall_distance < align:
                smooth = ((repulsion - wall_distance) / (align - repulsion)) * -1
                return (1 - smooth) * radial_vector + smooth * tangent_vector
            
    def distance_wall(self, x, y, vx, vy, kicktime, radius = 0.25):
        '''
        The function simulates the wall influence.
        x: Is the current x position of the focal fish.
        y: Is the current y position of the focal fish.
        vx: Is the velocity vector x of the focal fish.
        vy: Is the velocity vector y of the focal fish.
        kicktime: The current kicktime processed.
        Is the radius = 0.25 of the tank.
        '''
        # x, y are the initial coordinates (position of the fish at t = 0)
        # vx, vy are the velocity components in the x and y direction, respectively 
        # kicktime, is the time parameter that varies as the fish moves along the direction of its velocity vector 
        # x_fish, y_fish, path between closest point to wall and fish 
        x_fish = x 
        y_fish = y 
        print(f'x_fish: {x_fish}')
        print(f'y_fish: {y_fish}')

        # Calculate the distance between center (0,0) and fish position
        distance_to_origin = math.sqrt(x_fish**2 + y_fish**2)
        print(f'distance wall: {distance_to_origin}')
        print(f'radius: {radius}')

        # Calculate the closest point on the wall (radius = 0.25)
        x_wall = (x_fish  / distance_to_origin) * radius
        y_wall = (y_fish  / distance_to_origin) * radius


        # Repulsion Vector: Vector from the fish position (at kicktime) to the closest point on the wall

        vector_x = x_wall - x_fish
        vector_y = y_wall - y_fish

        # Push way away from the wall 
        radius_vector = np.array([vector_x, vector_y])
        radius_vector = radius_vector / np.linalg.norm(radius_vector)

        # Calculate the tangent - follow the wall 
        tangent = np.array([-y_wall, x_wall])
        tangent = tangent / np.linalg.norm(tangent)

        wall_distance = abs(radius - distance_to_origin)

        return radius_vector, tangent, x_wall, y_wall, wall_distance