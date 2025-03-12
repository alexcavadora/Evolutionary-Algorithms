from optFunc import AckleyFunction
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
        self.n = func.d
        self.history = []

    def init_population(self):
        ## Aqui se definen los rangos de la poblacion
        limit = 30
        self.pop = np.random.uniform(-limit, limit, (self.n_pop, self.n))

    def abc_return(self):   
        idx = np.random.choice(self.n_pop, 3, replace=False)
        return idx[0], idx[1], idx[2]
    
    def calculate_fitness_entropy(self):
        """Calcula la entropía basada en los valores de la función objetivo"""
        fitness_values = np.array([self.func.eval(ind) for ind in self.pop])
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
        entropy = self.calculate_fitness_entropy()
        if entropy < 1e-6:  # Umbral a ajustar
            print("Low entropy ")
            return False
            
        return True

    def get_best(self):
        fitness_values = np.array([self.func.eval(ind) for ind in self.pop])
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
        while self.stop_criteria(func_eval, n_gen):
            for i in range(self.n_pop):
                a,b,c = self.abc_return()
                q = np.copy(self.pop[i]) # ej q = [1,2,3,4,5]
                R = np.random.randint(0, self.n)
                for j in range(self.n):
                    if np.random.uniform(0,1) < self.CR or j == R:
                        q[j] = self.pop[a][j] + self.F * (self.pop[b][j] - self.pop[c][j])
                        # q(j) = x(j)a + F(x(j)b − x(j)c )
                q_eval =self.func.eval(q)
                pop_eval = self.func.eval(self.pop[i])
                func_eval += 2
                if q_eval < pop_eval:
                    self.pop[i] = q
            n_gen += 1
            best_fitness = float(self.func.eval(self.get_best()))
            self.history.append(best_fitness)
            if n_gen % 10 == 0:
                print(f"Generation {n_gen}, Best fitness: {best_fitness}")

        return self.get_best()
            



if __name__ == "__main__":
    x = [1e-16 for i in range(5)]
    ackley = AckleyFunction(x)
    algorithm = DE(ackley, n_pop= 10, n_gen= 1000, F = 0.8, CR = 0.9)
    best = algorithm.solve()
    print(f"Best solution: {best}")
    print(f"Best fitness: {ackley.eval(best)}")
    algorithm.plot()