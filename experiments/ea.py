import matplotlib.pyplot as plt
import numpy as np
import random
import csv
import math

from configs.config import Config

class EA:

    def __init__(self, config:Config):
        self.config = config

    def initialization(self, population_size, num_dimensions):
        """
        Initialize the starting population with random individuals.
        Each gene of an individual corresponds a dimension in the function
        param: population_size number of individuals in the population
        num_dimension: the number of dimensions
        """
        x = np.random.uniform(-1, 1, size = (population_size, num_dimensions)) # each individual is a vector

        # print(f'Initialization x: {x}')
        return x
    
    def objective_function(self, bias_flag, bias_func, bearing_file, data_dict, mode, dim, loss, c = None, focal_agent = 3, experiment_id = 151):  
        """
        Computes the loss between predicted and actual velocities for a focal fish.
        bias_flag: bool -> If bias_flag is True = dim 6.
        bias_func: Function to compute the bias vector.
        bearing_file: Dictionary containing the sorted information.
        data_dict: Dictionary with all relevant information.
        mode: String. Mode used in the bias function -> wall_zone.
        dim: Integer value - Dimension of the weights.
        loss: String. Type of loss function['cosine', 'mse_kicktime', mse_trajectory]
        c: [None] weights for the vector. Initially None, picked later at random - array.
        focal_agent: Integer value. Agent selected to model its behaviour - replace.
        experiemnt_id: Integer value. Experiment filter. 
        returns: float - Fitness - loss value.
        """
        predicted_positions_list = []
        actual_positions_list = []

        # Initalize weight vector c if not provided
        if c is None:
            c = np.random.rand(dim)

        # weights
        alpha, beta, gamma, delta, epsilon = c[0], c[1], c[2], c[3], c[4]
        bias = c[5]
        weights = np.array([alpha, beta, gamma, delta, epsilon, bias])

        # if bias = False -> dime 5 -> bias_zero
        if not bias_flag:
            weights[5] = 0.0
        
        # Normalise the weights using L1 norm
        if np.count_nonzero(weights) > 0:
            norm = np.linalg.norm(weights, ord = 1)
            weights = weights / norm

        kicktimes = []
        mse_loss = 0.0
        
        # Extract all kicktimes
        for key in bearing_file:
            kicktimes.append(key)

        # Get the initalial velocity of the focal agent at kicktime 0
        for agent_id, vx, vy in bearing_file[kicktimes[0]]:
            if agent_id == focal_agent:
                predicted_velocity = np.array([vx, vy])
                break
        

        # Inital position at timestep 0 of the focal agent 
        first_kicktime = np.min(kicktimes)
        initial_x = data_dict[first_kicktime]['x_coords'][0]  # focal agent is at position 0
        initial_y = data_dict[first_kicktime]['y_coords'][0]

        predicted_position = np.array([initial_x, initial_y])
        # print(predicted_position)

        # Iterate over kicktime
        for i in range(len(kicktimes)-1):

            kicktime = kicktimes[i]
            next_kicktime = kicktimes[i + 1]
            
            velocities = bearing_file[kicktime]
            next_velocities = bearing_file[next_kicktime]

            updated_velocity = []
            
            # Construct velocity array - use the predicted velocity for the focal agent and actual data for the other
            for j, (agent_id, vx, vy) in enumerate(next_velocities):
                # Extract the focal agents data - modify - use the precited once
                if j == 0:
                    updated_velocity.append((agent_id, predicted_velocity[0], predicted_velocity[1]))
                else:
                    updated_velocity.append((agent_id, vx, vy))
            
            vx_total = 0.0
            vy_total = 0.0   
            
            # Vector summation
            for j, (agent_id, vx, vy) in enumerate(updated_velocity):
                weight = weights[j]
            
                vx_total += vx * weight  
                vy_total += vy * weight 
            
            # if bias = True -> add a 6 vector
            if bias_flag:
                bias = bias_func(data_dict, kicktime, predicted_velocity, mode)
                vx_total += bias[0] * weights[5]  # Multiply by the bias agent's weight
                vy_total += bias[1] * weights[5]  # Multiply by the bias agent's weight
            
            predicted_velocity = np.array([vx_total, vy_total]) 

            # Choose a loss function
            # cosine - similarity between two fish
            # mse kicktime - Eucledean Distance at each kicktime
            # mse trajectory - Eucleadean Distance for the whole trajectory summed up

            if loss == 'cosine':
                actual_velocity = np.array([
                data_dict[next_kicktime]['vx_list'][0],
                data_dict[next_kicktime]['vy_list'][0]
            ])

                dotproduct = np.dot(actual_velocity, predicted_velocity)

                magnitude_a = np.linalg.norm(actual_velocity)
                magnitude_b = np.linalg.norm(predicted_velocity)

                cosine_similarity = dotproduct / (magnitude_a * magnitude_b)

                mse_loss += 1 - np.round(cosine_similarity, 4)
            
            else:

                if loss == 'mse_kicktime':
                    # per kicktime mse loss
                    updated_x = data_dict[kicktimes[i]]['x_coords'][0]  # focal agent is at position 0
                    updated_y = data_dict[kicktimes[i]]['y_coords'][0]

                    predicted_position = np.array([updated_x, updated_y])
                    predicted_position += predicted_velocity
                
                elif loss == 'mse_trajectory':
                    # trajectory mse loss
                    predicted_position += predicted_velocity


                actual_position = np.array([
                    data_dict[next_kicktime]['x_coords'][0],
                    data_dict[next_kicktime]['y_coords'][0]
                ])

                # Compute the loos - function - Eucleadean Distance
                delta_x = predicted_position[0] - actual_position[0]
                delta_y = predicted_position[1] - actual_position[1]
                distance = np.sqrt(delta_x**2 + delta_y**2)
                
                # Squared Error
                mse_loss += distance** 2

        return mse_loss
    
    def evaluation(self, x, objective_function, bias_flag, bias_func, orientation_file, data_dict, mode, dim, loss): 
        """Evaluate the fitness of the population members"""

        fitness = []

        # iterating over each member of the population and evaluating each members fitness
        for i in range(len(x)):
            mse_loss = objective_function(bias_flag = bias_flag, bias_func = bias_func, orientation_file= orientation_file, data_dict = data_dict, mode = mode, dim =dim, loss = loss, c = x[i])
            fitness.append(mse_loss)

        fitness = np.array(fitness) #converting the stored list into an numpy array
        return fitness
    
    def mutation(self, x, mutation_rate):
        """
        Apply mutation by adding small Gaussian noise to each element.
    
        """

        for individual in x: 
            for i in range(len(individual)):
                ran = np.random.uniform(0.0, 1) - 0.5
                individual[i] = individual[i] + mutation_rate * ran

        x = np.clip(x, -1, 1)

        return x
    

    def crossover(self, x_parents, p_crossover, objective_function,  bias_flag, bias_func, orientation_file, data_dict, mode, dim, loss):
        """Perform crossover to create offsprings."""

        fitness = self.evaluation(x_parents, objective_function, bias_flag, bias_func, orientation_file, data_dict, mode, dim, loss) # Evaluate fitness of parents
        best_fitness = np.argsort(fitness) # Sorted based on fitness to find the minimum
        num_parents = len(x_parents)
        num_dimensions = len(x_parents[0])
        offspring = np.empty_like(x_parents) # Return an empty array with shape and type of input -> x_parents.

        for i in range(0, num_parents, 2): # iterate over pairs of parent
            parent1 = x_parents[best_fitness[i]] # select the best fitness
            parent2 = x_parents[best_fitness[i + 1]] # select the second parent with the next best fitness

            if np.random.rand() < p_crossover: # check if cross_over should occur
                crossover_point = np.random.randint(1, num_dimensions) # crossover point randomized

                # Create offbrings by concatenating both strings of parent determined by a random number
                offspring[i] = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                offspring[i + 1] = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))

            else: # cross_over did not occur, keep the best fitness as offspring
                offspring[i] = x_parents[best_fitness[i]]
                offspring[i + 1] = x_parents[best_fitness[i + 1]]
        
        # print(f'x_parents: crossover: {x_parents}')
        # print(f'offspring: crossover: {offspring}')

        return offspring
    
    def parent_selection(self, x, f, k):
        """Select parents for the next generation"""

        x_array = np.array(x)
        f_array = np.array(f)

        matching_pool_x = []
        matching_pool_f = []

        while len(matching_pool_x) < len(x_array):
            idx = np.random.choice(len(x_array), k, replace = False)
            fitness = f[idx]

            best_idx = idx[np.argmin(fitness)]

            matching_pool_x.append(x_array[best_idx])
            matching_pool_f.append(f_array[best_idx])

        return np.array(matching_pool_x), np.array(matching_pool_f)
    
    def survivor_selection(self, x, f, x_offspring, f_offspring):
        """Select the survivors, for the population of the next generation"""

        #Concatenate parents ans their offspring
        combined_population = np.concatenate((x, x_offspring))

        # Concatenate fitness of parents and their offspring
        combined_fitness = np.concatenate((f, f_offspring))

        # Sorting the fitness
        best_fitness_index = np.flip(np.argsort(combined_fitness))[::-1]


        # new poulation based on the best fitness of parent and offspring
        x = combined_population[best_fitness_index[:len(x)]]
        # print(f'Combined_Population: {x}')

        f = combined_fitness[best_fitness_index[:len(x)]]
        # print(f'Combined_Fitness: {f}')
        return x, f
    
    def ea(
        self,
        # hyperparameters of the algorithm
        population_size,
        max_fit_evals,  # Maximum number of evaluations
        p_crossover,  # Probability of performing crossover operator
        m_rate,
        k,   # mutation rate
        mode,
        dimensions,  # number of dimensions
        bias_flag, 
        bias_func,
        orientation_file, 
        data_dict, 
        loss,
        objective_function,  # objective function to be minimized

    ):
    
        # Calculate the maximum number of generations
        # Maximum number of function evaluations should be the same independent of the population size
        x = self.initialization(population_size, dimensions)
        print(f"Initial population shape: {x.shape}")  # Should be (population_size, 6)
        
        max_generations = 100 
        
        # max_generations = int(max_fit_evals / population_size)  # DO NOT CHANGE
        # print(f'max generation: {max_generations}')
        
        ################################################################
        # PLEASE FILL IN
        # x = initialization(population_size, dimensions)
        f = self.evaluation(x, objective_function, bias_flag, bias_func, orientation_file, data_dict, mode, dimensions, loss)
        ################################################################

        # Find the best individual and append to a list to keep track in each generation
        idx = np.argmin(f)
        x_best = [x[idx]]
        f_best = [f[idx]]
        weights_best = [x_best[0]]


        # Loop over the generations
        for gen in range(max_generations - 1):
            # Perform the EA steps
            x_parents, f_parents = self.parent_selection(x, f, k)
            # print(len(x), len(x_parents))

            x_offspring = self.crossover(x_parents, p_crossover, objective_function, bias_flag, bias_func, orientation_file, data_dict, mode, dimensions, loss)
            x_offspring = self.mutation(x_offspring, mutation_rate= m_rate)
            f_offspring = self.evaluation(x_offspring, objective_function, bias_flag, bias_func, orientation_file, data_dict, mode, dimensions, loss)
            x,f= self.survivor_selection(x, f, x_offspring, f_offspring)
            
            ################################################################
            
            # PLEASE FILL IN

            ################################################################
            

            # Find the best individual in current generation and add to the list
            idx = np.argmin(f)
            xi_best = x[idx]
            fi_best = f[idx]

            # if fi_best < f_best[-1]:
            x_best.append(xi_best)
            f_best.append(fi_best)
                

        return x_best, f_best # return the best solution and fitness in each generation