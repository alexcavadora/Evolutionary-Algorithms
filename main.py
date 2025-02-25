import numpy as np

from table_gen import table


class GA:
  def __init__(self, n, npop, feval, ngen, pmut, pcross, ptr_func_init=None):
    self.npop = npop
    self.feval = feval
    self.ngen = ngen
    self.pmut = pmut
    self.pcross = pcross
    self.ptr_func_init = ptr_func_init
    self.pop = [table(n) for _ in range(npop)]
    self.pop_fit = np.zeros(npop)
    self.pop_select = [table(n) for _ in range(npop)]
    self.offspring = [table(n) for _ in range(npop)]
    
    # Initialize the population
  
  def solve(self):
    pass


GA(8, 10, 100, 100, 0.1, 0.9)
