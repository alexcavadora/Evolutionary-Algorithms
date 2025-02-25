import numpy as np

from table_gen import table
from table_gen import mask


class GA:
  def __init__(self, n, npop, feval, ngen, pmut, pcross, ptr_func_init=None):
    self.npop = npop
    self.feval = feval
    self.ngen = ngen
    self.pmut = pmut
    self.pcross = pcross
    self.ptr_func_init = ptr_func_init
    self.pop = [table(n) for _ in range(npop)]
    for i in range(npop):
      self.pop[i].generate_random_table()
    self.pop_fit = np.zeros(npop)
    self.pop_select = [table(n) for _ in range(npop)]
    self.offspring = [table(n) for _ in range(npop)]
    self.mask = mask(n)
    self.mask.init_mask_array()
    
    # Initialize the population
  
  @staticmethod
  def popcount(x):
    """
    Cuenta el número de bits activos en un entero usando el algoritmo de Brian Kernighan.
    Este método es más eficiente que contar bit por bit.
    """
    count = 0
    while x:
        x &= (x - 1)  # Elimina el bit 1 menos significativo
        count += 1
    return count

  @staticmethod
  def count_queens_collisions(self, individual):
  # Contar colisiones y reinas
    collisions = 0
    queen_count = 0
    n = individual.n

    for row in range(n):
        for col in range(n):
            bit_index = row * n + col
            idx = bit_index // 64
            offset = (bit_index % 64)
            if np.bitwise_and((individual.table[idx]), np.uint64(1 << offset)):
                queen_count += 1
                # Revisar colisiones
                ind_masked = individual.copy()
                self.mask.apply_mask(ind_masked, row, col)
  
                for i in range(len(ind_masked.table)):
                    number = int(ind_masked.table[i].item())
                    collisions += self.popcount(number)

    return queen_count, collisions

  def evaluate_solution(self, individual):
    queen_count, collisions = self.count_queens_collisions(self, individual)
    # print(f'Collisions: {collisions}, Queens: { queen_count}')
    # return 1.0 / (queen_count + collisions) if (queen_count + collisions) > 0 else 0
    aux = (1.0 / n) + collisions
    return 1.0 / (aux + 1e-6)
  
  def fitness_eval(self):
    for i in range(0,self.npop):
      self.pop_fit[i] = self.evaluate_solution(self.pop[i])
    
    # Encontrar y preservar el mejor individuo
    best_idx = np.argmax(self.pop_fit)
    self.pop_select[0] = self.pop[best_idx].copy()

  def select_pop(self):
    for i in range(1,self.npop):
      # Select two random individuals
      ii, jj = np.random.choice(self.npop, 2, replace=False)
      
      # Comparar fitness y seleccionar el mejor
      if self.pop_fit[ii] < self.pop_fit[jj]:
          self.pop_select[i] = self.pop[jj].copy()  # El de mayor fitness
      else:
          self.pop_select[i] = self.pop[ii].copy()  # El de mayor fitness

  def one_point_crossover(self, idx1, idx2):
    """
    Realiza el crossover de un punto entre dos individuos
    Args:
        idx1: índice del primer padre
        idx2: índice del segundo padre
    """
    n = self.pop[0].n
    # Punto de cruce aleatorio
    cross_point = np.random.randint(0, n*n)
    
    # Crear copias de los padres
    offspring1 = self.pop_select[idx1].copy()
    offspring2 = self.pop_select[idx2].copy()
    
    # Realizar el crossover bit a bit
    for bit in range(cross_point, n*n):
        idx = bit // 64
        offset = int(bit % 64)
        
        # Obtener los bits de ambos padres
        bit1 = int(offspring1.table[idx].item()) & (1 << offset)
        bit2 = int(offspring2.table[idx].item()) & (1 << offset)
        
        # Intercambiar bits
        aux = np.uint64(1 << offset)
        if bit1:
            offspring2.table[idx] |= aux
        else:
            offspring2.table[idx] &= ~aux
            
        if bit2:
            offspring1.table[idx] |= aux
        else:
            offspring1.table[idx] &= ~aux
    
    # Guardar descendientes
    self.offspring[idx1] = offspring1
    self.offspring[idx2] = offspring2

  def crossover(self):
    for i in range(0,self.npop,2):
      u = np.random.rand() # Random number between 0 and 1
      if u < self.pcross:
        # Realizar crossover de un punto
        self.one_point_crossover(i, i+1)
      else:
        # Copiar directamente los padres
        self.offspring[i] = self.pop_select[i].copy()
        self.offspring[i+1] = self.pop_select[i+1].copy()

  def mutate(self):
    """
    Aplica mutación bit a bit a toda la población offspring
    usando XOR para invertir bits con probabilidad pmut
    """
    n = self.pop[0].n
    n_bits = n * n
    
    for i in range(1,self.npop):
        for bit in range(n_bits):
            u = np.random.rand() # Random number between 0 and 1
            if u < self.pmut:
                idx = bit // 64
                offset = int(bit % 64)
                # Aplicar XOR para invertir el bit
                self.offspring[i].table[idx] ^= np.uint64(1 << offset)
                ## ^= operador de asignación de bits XOR

  def union(self):
    """
    Reemplaza la población actual con la descendencia
    """
    for i in range(self.npop):
        self.pop[i] = self.offspring[i].copy()
  
  def solve(self):
    """
    Ejecuta el algoritmo genético completo
    """
    k_gen = 0
    best_fitness = float('-inf')
    best_solution = None
    q, c = [0,10]
    while k_gen < self.ngen:  # Criterio de parada: número de generaciones
        k_gen += 1
        
        # Evaluar población actual
        self.fitness_eval()
        
        # Guardar la mejor solución encontrada
        current_best = np.argmax(self.pop_fit)
        if self.pop_fit[current_best] > best_fitness:
            best_fitness = self.pop_fit[current_best]
            best_solution = self.pop[current_best].copy()
            
        # Proceso evolutivo
        self.select_pop()    # Selección
        self.crossover()     # Cruce
        self.mutate()        # Mutación
        self.union()         # Actualizar población
        
        q,c = self.count_queens_collisions(self, best_solution)

        # Opcional: mostrar progreso
        print(f"Generación {k_gen}: Mejor fitness = {best_fitness:2f} - Queens = {q} - Colisiones = {c}")
    
    return best_solution, best_fitness
    


if __name__ == "__main__":
  # Parameters
  n = 10           # nxn board
  npop = 100      # population size 
  ngen = 100     # number of generations
  pmut = 0.1     # mutation probability
  pcross = 0.85    # crossover probability
  
  # Create and run GA
  ga = GA(n, npop, None, ngen, pmut, pcross)
  best_solution, best_fitness = ga.solve()
  
  # Print results
  print("\nBest solution found:")
  best_solution.plot_chessboard()
  print(f"Fitness: {best_fitness}")

