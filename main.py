import random
import numpy as np
import matplotlib.pyplot as plt

def generate_individual(n):
    return random.sample(range(n), n)

def fitness(individual):
    n = len(individual)
    diag1 = np.zeros(2 * n - 1, dtype=int)
    diag2 = np.zeros(2 * n - 1, dtype=int)
    
    for row in range(n):
        diag1[individual[row] - row + n - 1] += 1
        diag2[individual[row] + row] += 1
    
    return sum(v * (v - 1) // 2 for v in np.concatenate((diag1, diag2)) if v > 1)

def tournament_selection(population, fitness_vals, tournament_size=3):
    return [min(random.sample(population, tournament_size), key=lambda ind: fitness(ind)) for _ in range(len(population))]

def order_crossover(parent1, parent2):
    n = len(parent1)
    c1, c2 = sorted(random.sample(range(n), 2))
    
    def create_child(parent, other):
        child = [None] * n
        child[c1:c2+1] = parent[c1:c2+1]
        remaining = [gene for gene in other if gene not in child]
        child = [remaining.pop(0) if v is None else v for v in child]
        return child
    
    return create_child(parent1, parent2), create_child(parent2, parent1)

def swap_mutation(individual, mutation_prob=0.1):
    if random.random() < mutation_prob:
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return individual

def ga_n_queens(n, population_size=100, generations=1000, mutation_prob=0.1, crossover_prob=0.8):
    population = [generate_individual(n) for _ in range(population_size)]
    best_individual, best_fitness = min(((ind, fitness(ind)) for ind in population), key=lambda x: x[1])
    
    for gen in range(generations):
        fitness_vals = [fitness(ind) for ind in population]
        current_best = min(zip(population, fitness_vals), key=lambda x: x[1])
        
        if current_best[1] < best_fitness:
            best_individual, best_fitness = current_best
        
        print(f"Generation {gen+1}: best fitness = {best_fitness}")
        if best_fitness == 0:
            break
        
        selected = tournament_selection(population, fitness_vals)
        new_population = []
        
        for i in range(0, population_size, 2):
            parent1, parent2 = selected[i], selected[(i+1) % population_size]
            offspring = order_crossover(parent1, parent2) if random.random() < crossover_prob else (parent1, parent2)
            new_population.extend([swap_mutation(child, mutation_prob) for child in offspring])
        
        population = new_population[:population_size]
    
    return best_individual, best_fitness

def plot_chessboard(solution):
    n = len(solution)
    board = np.zeros((n, n))
    board[::2, 1::2] = board[1::2, ::2] = 1
    
    fig, ax = plt.subplots()
    ax.imshow(board, cmap='gray', extent=(0, n, 0, n))
    for row, col in enumerate(solution):
        ax.text(col + 0.5, n - row - 0.5, '♛', ha='center', va='center', color='#FFD700', fontsize=20)
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_yticklabels(range(n, 0, -1))
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.invert_yaxis()
    plt.show()

solution, solution_fitness = ga_n_queens(200, population_size=100, generations=10000, mutation_prob=0.15, crossover_prob=0.9)

print("\nSolution found:", solution)
print("Conflicts:", solution_fitness)
if solution_fitness == 0:
    plot_chessboard(solution)
