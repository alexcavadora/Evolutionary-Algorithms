import numpy as np
from pid_function import PIDFunction
from differential_evolution import differential_evolution

def main():
    # Define the parameters for the PIDFunction
    x0 = [0, 0]  # Initial position
    t_end = 5  # End time for the simulation
    del_t = 0.05  # Time step for the simulation
    x = [10, 10]  # Amplitude of the trajectory in X and Y
    w = [1, 1]  # Frequency of the trajectory in X and Y
    teta = [45, 45]  # Angles for the arm
    sides = [3, 3]  # Lengths of the arm sides  

    # Create an instance of the PIDFunction
    pid_function = PIDFunction(x0, t_end, del_t, x, w, teta, sides)

    # Set parameters for the differential evolution
    population_size = 30  # Reduce the population size
    generations = 100  # Reduce the number of generations
    mutation_factor = 0.7
    crossover_probability = 0.7
    bounds = [(0, 1), (0, 0.1), (0, 0.01)]  # Example bounds for PID parameters

    # Initialize the Differential Evolution algorithm
    best_solution = differential_evolution(
        pid_function, 
        pop_size=population_size, 
        dim=3, 
        bounds=bounds, 
        mutation_factor=mutation_factor, 
        crossover_rate=crossover_probability, 
        max_generations=generations
    )

    # Print the best solution found
    print(f'Best solution (PID parameters): {best_solution}')

    # Set the best PID parameters to the PIDFunction instance
    pid_function.kc, pid_function.ki, pid_function.kd = best_solution

    # Plot the trajectory and position to verify the particle follows the desired trajectory
    pid_function.plot_trajectory()
    pid_function.plot_position()

if __name__ == "__main__":
    main()