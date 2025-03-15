import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from pid_function import PIDFunction

def initialize_population(pop_size, dim, bounds):
    population = np.random.rand(pop_size, dim)
    for i in range(dim):
        population[:, i] = bounds[i][0] + population[:, i] * (bounds[i][1] - bounds[i][0])
    return population

def evaluate_fitness(population, pid_function):
    fitness = [pid_function.evaluate(individual) for individual in population]
    return np.array(fitness)

def select_parents(population, fitness):
    parents_indices = np.random.choice(len(population), size=3, replace=False)
    return population[parents_indices]

def crossover(parent1, parent2, crossover_rate):
    mask = np.random.rand(len(parent1)) < crossover_rate
    trial = np.where(mask, parent2, parent1)
    return trial

def mutate(parents, mutation_factor, bounds):
    a, b, c = parents
    mutant = np.clip(a + mutation_factor * (b - c), 0, 1)
    for i in range(len(mutant)):
        mutant[i] = bounds[i][0] + mutant[i] * (bounds[i][1] - bounds[i][0])
    return mutant

def differential_evolution(pid_function, pop_size, dim, bounds, mutation_factor, crossover_rate, max_generations):
    population = initialize_population(pop_size, dim, bounds)
    best_positions = []
    best_fitness = []

    for generation in range(max_generations):
        fitness = evaluate_fitness(population, pid_function)
        new_population = []
        for i in range(pop_size):
            parents = select_parents(population, fitness)
            mutant = mutate(parents, mutation_factor, bounds)
            trial = crossover(population[i], mutant, crossover_rate)
            trial_fitness = pid_function.evaluate(trial)
            if trial_fitness < fitness[i]:
                new_population.append(trial)
            else:
                new_population.append(population[i])
        population = np.array(new_population)
        fitness = evaluate_fitness(population, pid_function)
        best_index = np.argmin(fitness)
        best_positions.append(population[best_index])
        best_fitness.append(fitness[best_index])
        if generation % 10 == 0:
            print(f'Generation {generation}: Best Fitness = {fitness[best_index]}')

    best_index = np.argmin(fitness)
    best_solution = population[best_index]

    # Plot the best positions over generations
    plt.figure(figsize=(8, 6))
    plt.plot(range(max_generations), best_fitness, label='Best Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Best Fitness Over Generations')
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_solution