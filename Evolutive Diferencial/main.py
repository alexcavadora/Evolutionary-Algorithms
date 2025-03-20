import numpy as np
from pid_function import PIDFunction
from differential_evolution import differential_evolution

def main():
    # Define the parameters for the PIDFunction
    x0 = [0, 0]  # Initial position
    t_end = 10  # End time for the simulation
    del_t = 0.1  # Smaller time step for better accuracy
    x = [0.5, 0.5]  # Amplitude of the trajectory in X and Y
    w = [1, 1]  # Frequency of the trajectory in X and Y
    teta = [45, 45]  # Angles for the arm
    sides = [1, 1]  # Lengths of the arm sides  

    # Create an instance of the PIDFunction
    pid_function = PIDFunction(x0, t_end, del_t, x, w, teta, sides)

    # Set parameters for the differential evolution
    population_size = 100  # Increase population size for better exploration
    generations = 100  # Increase generations for better convergence
    mutation_factor = 0.5  # Standard mutation factor
    crossover_probability = 0.7
    # Wider bounds for PID parameters to allow better exploration
    bounds = [(0, 2), (0, 0.5), (0, 0.1)]  # Bounds for Kp, Ki, Kd

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