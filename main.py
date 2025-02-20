import numpy as np

def generate_n_numbers_of_n_bytes(n):
  return [bytes(n) for _ in range(n)]

class GA:
  def __init__(self, n, npop, feval, ngen, pmut, pcross, ptr_func_init=None):
    self.npop = npop
    self.feval = feval
    self.ngen = ngen
    self.pmut = pmut
    self.pcross = pcross
    self.ptr_func_init = ptr_func_init
    self.pop = [generate_n_numbers_of_n_bytes(n) for _ in range(npop)]
    self.pop_fit = np.zeros(npop)
    self.pop_select = [generate_n_numbers_of_n_bytes(n) for _ in range(npop)]
    self.offspring = [generate_n_numbers_of_n_bytes(n) for _ in range(npop)]
    
    # Initialize the population
  
  def solve(self):
    pass