from optFunc import PIDFunction
import numpy as np
import matplotlib.pyplot as plt

class DE:
    def __init__(self,func,n_pop, n_gen, F = 0.8, CR = 0.9):
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.pop = None
        self.F = F
        self.CR = CR
        self.func = func
        self.n = 3
        self.history = []

    def init_population(self):
        ## Aqui se definen los rangos de la poblacion
        limit = 15
        # self.pop = np.random.uniform(0, limit, (self.n_pop, self.n))
        self.pop = np.zeros((self.n_pop, self.n))
        for i in range(self.n_pop):
            # Kp: proporcional (valores más altos)
            self.pop[i, 0] = np.random.uniform(0.5, 15)
            
            # Ki: integral (valores intermedios)
            self.pop[i, 1] = np.random.uniform(0.01, 2.0)
            
            # Kd: derivativo (valores más pequeños)
            self.pop[i, 2] = np.random.uniform(0.01, 5.0)

    def abc_return(self):   
        idx = np.random.choice(self.n_pop, 3, replace=False)
        return idx[0], idx[1], idx[2]
    
    def calculate_fitness_entropy(self):
        """Calcula la entropía basada en los valores de la función objetivo"""
        fitness_values = np.array([self.func.evaluate(ind) for ind in self.pop])
        # Si la desviación estándar es pequeña, la población ha convergido
        return np.std(fitness_values)

    def stop_criteria(self, func_eval, gen_count=0):
        # Criterio por número de evaluaciones
        if func_eval >= 100000:
            print("Max function evaluations reached")
            return False
        
        # Criterio por número de generaciones
        if gen_count >= self.n_gen:
            print("Max generations reached")
            return False
        
        # Criterio por entropía (diversidad de valores fitness)
        # entropy = self.calculate_fitness_entropy()
        # if entropy < 1e-6:  # Umbral a ajustar
        #     print("Low entropy ")
        #     return False
            
        return True

    def get_best(self):
        fitness_values = np.array([self.func.evaluate(ind) for ind in self.pop])
        best_idx = np.argmin(fitness_values)
        return self.pop[best_idx]

    def plot(self):
        plt.style.use("ggplot")
        # Intercambia el orden de los argumentos para que el fitness esté en el eje Y
        plt.plot(range(len(self.history)), self.history)
        plt.title("Fitness over generations")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.show()

    def solve(self):
        self.init_population()
        # print(f"Initial population: {self.pop}")
        # print(f"Initial fitness: {self.func.eval(self.pop[0])}")
        func_eval = 0
        n_gen = 0
        self.best = self.pop[0]
        print("Inicializando algoritmo")
        while self.stop_criteria(func_eval, n_gen):
            for i in range(self.n_pop):
                a,b,c = self.abc_return()
                q = np.copy(self.pop[i]) # ej q = [1,2,3,4,5]
                R = np.random.randint(0, self.n)
                for j in range(self.n):
                    if np.random.uniform(0,1) < self.CR or j == R:
                        q[j] = self.pop[a][j] + self.F * (self.pop[b][j] - self.pop[c][j])
                        # q(j) = x(j)a + F(x(j)b − x(j)c )
                q_eval =self.func.evaluate(q)
                # print(f"Q: {q}, Q_eval: {q_eval}")
                pop_eval = self.func.evaluate(self.pop[i])
                # print(f"Pop: {self.pop[i]}, Pop_eval: {pop_eval}")
                func_eval += 2
                if q_eval < pop_eval:
                    self.pop[i] = q
                    # print(f'pop[i]: {self.pop[i]}')
            n_gen += 1
            # best_fitness = float(self.func.evaluate(self.get_best()))
            # self.history.append(best_fitness)
            # if n_gen % 10 == 0:
            print(f"Generation {n_gen}")
            # print(f"Generation {n_gen}, Best error: {best_fitness}")

        return self.get_best()
            



if __name__ == "__main__":
    
    # Define the parameters for the PIDFunction
    x0 = [0, 0]  # Initial position
    t_end = 30  # End time for the simulation
    del_t = 0.25  # Time step for the simulation
    x = [5, 5]  # Amplitude of the trajectory in X and Y
    w = [1, 3]  # Frequency of the trajectory in X and Y
    teta = [45, 45]  #Initial Angles for the arm
    sides = [3, 3]  # Lengths of the arm sides

    # Create an instance of the PIDFunction
    pid_function = PIDFunction(x0, t_end,del_t, x, w, teta, sides)
    pid_function.plot_trajectory()

    F = 0.8 # Factor de escala
    CR = 0.9 # Factor de cruce

    algorithm = DE(pid_function, n_pop= 10, n_gen= 10, F = 0.8, CR = 0.9)
    best = algorithm.solve()
    print(f"Best solution: {best}")
    print(f"Best error: {pid_function.evaluate(best)}")
    # algorithm.plot()
    # Plot the position
    pid_function.plot_position()