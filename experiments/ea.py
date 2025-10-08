import numpy as np
from typing import Tuple, Dict, Any

from configs.config import Config
from experiments.objective_function import ObjectiveFunctionEvaluator

class EA:
    def __init__(self, config:Config, data_dict:Dict):
        self.config = config
        self.data_dict = data_dict
        self.objective_function_evaluator = ObjectiveFunctionEvaluator(self.config, self.data_dict)

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
    
    def evaluation(self, x): 
        """Evaluate the fitness of the population members"""
        fitness = []

        # iterating over each member of the population and evaluating each members fitness
        for i in range(len(x)):
            mse_loss = self.objective_function_evaluator.objective_function(x[i])
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
    

    def crossover(self, x_parents, p_crossover):
        """Perform crossover to create offsprings."""

        fitness = self.evaluation(x_parents) # Evaluate fitness of parents
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
    
    def run(self):
        x = self.initialization(self.config.population_size, self.config.dimensions)
        print(f"Initial population shape: {x.shape}")  # Should be (population_size, 6)
        
        f = self.evaluation(x)

        # Find the best individual and append to a list to keep track in each generation
        idx = np.argmin(f)
        x_best = [x[idx]]
        f_best = [f[idx]]
        weights_best = [x_best[0]]

        # Loop over the generations
        for gen in range(self.config.num_generations - 1):
            # Perform the EA steps
            x_parents, f_parents = self.parent_selection(x, f, self.config.k)
            # print(len(x), len(x_parents))

            x_offspring = self.crossover(x_parents, self.config.p_crossover)
            x_offspring = self.mutation(x_offspring, mutation_rate=self.config.mutation_rate)
            f_offspring = self.evaluation(x_offspring)
            x,f= self.survivor_selection(x, f, x_offspring, f_offspring)

            # Find the best individual in current generation and add to the list
            idx = np.argmin(f)
            xi_best = x[idx]
            fi_best = f[idx]

            # if fi_best < f_best[-1]:
            x_best.append(xi_best)
            f_best.append(fi_best)

        return x_best, f_best # return the best solution and fitness in each generation