from Parser import SimulatedCircuit
from NodeClass import node
import random

class GA:

    def __init__(self, n_pop, n_gen, p_crossover, p_mutation):
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.pop = []
        self.pop_fitness = []
        self.best = None
        self.best_fitness = 1000
        self.pop_select = [0]*n_pop
        self.offspring = [0]*n_pop
        self.decoder = SimulatedCircuit("dec.blif")
        self.decoder.simulate()
        
    def init_pop(self):
        for i in range(self.n_pop):
            n = node()
            tree = n.rand_tree(4) # 4 is the depth of the tree
            self.pop.append(tree)

    def fitness_fun(self, tree):
        output_vector = []
        for input_vector in self.decoder.inputs_history: # Itera en todas las posibles entradas
            output = tree.evaluate(input_vector) # Evalua el arbol con la entrada
            output_vector.append(output) # Almacena la salida

        error = self.decoder.compare_outputs(output_vector)
        # depht = tree.get_depth()
        return error
    
    def evaluate_pop(self):
        self.pop_fitness = []
        for tree in self.pop:
            fitness = self.fitness_fun(tree)
            self.pop_fitness.append(fitness)
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best = tree
        self.pop_select[0] = self.best.copy()

    def tournament_selection(self):
        ## Funciona bien pero podria considerar individuos repetidos
        idxs = random.sample(range(1,self.n_pop), 2)
        if self.pop_fitness[idxs[0]] < self.pop_fitness[idxs[1]]:
            return self.pop[idxs[0]]
        else:
            return self.pop[idxs[1]]
    
    def roulette_selection(self):
        total = sum(self.pop_fitness)
        pick = random.uniform(0, total)
        current = 0
        for i in range(self.n_pop):
            current += self.pop_fitness[i]
            if current > pick:
                return self.pop[i]
    
    def select_pop(self):
        for i in range(1,self.n_pop):
            if random.uniform(0, 1) < 0.5: ## Se puede cambiar para que use una u otra
                self.pop_select[i] = self.tournament_selection()
            else:
                self.pop_select[i] = self.roulette_selection()

    def crossover(self):
        for i in range(0,self.n_pop,2):
            if random.uniform(0, 1) < self.p_crossover:
                # Swap the trees
                self.offspring[i], self.offspring[i+1] = self.pop_select[i].swap(self.pop_select[i+1])
            else:
                self.offspring[i] = self.pop_select[i]
                self.offspring[i+1] = self.pop_select[i+1]
    
    def mutation(self):
        for i in range(self.n_pop):
            if random.uniform(0, 1) < self.p_mutation:
                self.offspring[i].mutation(5)

    def solve(self):
        self.init_pop()
        for i in range(self.n_gen):
            self.evaluate_pop()
            self.select_pop()
            self.crossover()
            self.mutation()
            self.pop = self.offspring.copy()
            print(f"Generation: {i}, Best fitness: {self.best_fitness}")
            if self.best_fitness == 0:
                break

        print(f"Best fitness: {self.best_fitness}")


if __name__ == "__main__":

    POPULATION = 80
    GENERATIONS = 100
    CROSSOVER = 0.75
    MUTATION = 0.1


    algorithm = GA(POPULATION, GENERATIONS, CROSSOVER, MUTATION);
    algorithm.solve();
    algorithm.best.visualize();
    print(algorithm.best)