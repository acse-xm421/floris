import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize

from floris.tools import FlorisInterface
from floris.tools.visualization import (
    calculate_horizontal_plane_with_turbines,
    visualize_cut_plane,
)

# Initialize the FLORIS interface fi
file_dir = os.path.dirname(os.path.abspath(__file__))
fi = FlorisInterface('inputs/gch.yaml')

# Setup 1 wind directions with 1 wind speed and frequency distribution
wind_directions = [0.0,]
wind_speeds = [8.0,]
# Shape frequency distribution to match number of wind directions and wind speeds
freq = (
    np.abs(
        np.sort(
            np.random.randn(len(wind_directions))
        )
    )
    .reshape( ( len(wind_directions), len(wind_speeds) ) )
)
freq = freq / freq.sum()
print(freq)
fi.reinitialize(wind_directions=wind_directions, wind_speeds=wind_speeds)

D = 126.0 # rotor diameter for the NREL 5MW
# boundary = [(0.0, 0.0), (0.0, 40*D), (40*D, 40*D), (40*D, 0.0), (0.0, 0.0)]

# Define the wind farm parameters
N_turbines = 4

# Define the GA parameters
population_size = 2
chromosome_length = 100
generations = 5
mutation_rate = 0.01

rows = np.square(chromosome_length)
cols = rows
side_length = 4*D


# Define the fitness function
def evaluate_fitness(chromosome):
    layout = get_layout_from_chromosome(chromosome)
    layout_x = layout[:, 0]
    layout_y = layout[:, 1]
    fi.reinitialize(layout_x=layout_x, layout_y=layout_y)

    # fi.farm.set_boundaries(boundary)
    fi.calculate_wake()

    # Calculate AEP
    aep = fi.get_farm_AEP(freq=freq) / 1e6

    return -aep  # Negative sign for maximization problem

def get_i_n_j(index):
    j = index // cols
    i = index % cols
    return i, j

# Function to decode chromosome into layout
def get_layout_from_chromosome(chromosome):
    layout = np.zeros((N_turbines, 2))
    index = 0
    n_turbine = 0

    while 1:
        if n_turbine >= N_turbines - 1 or index >= 100:
            break
        if chromosome[index]==1:
            i, j = get_i_n_j(index)
            layout[n_turbine, 0] = i*side_length+side_length/2
            layout[n_turbine, 1] = j*side_length+side_length/2
            n_turbine += 1
        index += 1

    return layout

# Function to encode chromosome from random number
def get_chromosome_from_num(population_num):
    chromosome = np.zeros((population_size, chromosome_length))
    count = 0
    for indiv in population_num:
        for turbine in indiv:
            chromosome[count, turbine] = 1
        count += 1

    return chromosome

# GA initialization
def initialize_population():
    population_num = np.random.randint(chromosome_length, size=(population_size, N_turbines))
    population = get_chromosome_from_num(population_num)
    return population

# GA mutation #wrong
def mutate(chromosome):
    for i in range(chromosome_length):
        if np.random.random() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]  # Flip bit
    return chromosome

# GA crossover
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, chromosome_length)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

# GA selection (tournament selection)
def selection(population, fitness_values):
    indices = np.random.choice(population_size, size=2, replace=False)
    if fitness_values[indices[0]] > fitness_values[indices[1]]:
        return population[indices[0]]
    else:
        return population[indices[1]]
    
# GA selection (Roulette Selection)


# Main GA loop
def run_ga():
    population = initialize_population()

    for generation in range(generations):
        fitness_values = np.zeros(population_size)
        for i in range(population_size):
            chromosome = population[i]
            # chromosome = mutate(chromosome)
            # population[i] = chromosome
            fitness_values[i] = evaluate_fitness(chromosome)

        new_population = np.zeros_like(population)
        for i in range(population_size // 2):
            parent1 = selection(population, fitness_values)
            parent2 = selection(population, fitness_values)
            child1, child2 = crossover(parent1, parent2)
            new_population[i * 2] = child1
            new_population[i * 2 + 1] = child2

        # for i in range(population_size):
        #     chromosome = population[i]
        #     chromosome = mutate(chromosome)
        #     population[i] = chromosome

        population = new_population

    best_chromosome = population[np.argmax(fitness_values)]
    best_layout = get_layout_from_chromosome(best_chromosome)
    best_fitness = evaluate_fitness(best_chromosome)

    return best_layout, -best_fitness  # Return layout and positive AEP

# Run the GA
best_layout, best_aep = run_ga()

# Print the results
print("Best Layout:")
print(best_layout)
print("Best AEP:", best_aep)
