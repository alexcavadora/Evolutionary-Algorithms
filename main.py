import numpy as np

class GA:
  def __init__(self, n, npop, feval, ngen, pmut, pcross, ptr_func_init=None):
    self.npop = npop
    self.feval = feval
    self.ngen = ngen
    self.pmut = pmut
    self.pcross = pcross
    self.ptr_func_init = ptr_func_init
    self.pop = np.zeros((self.npop, n * n))
    self.pop_fit = np.zeros((self.npop, 1))
    self.pop_select = np.zeros((self.npop, n * n))
    self.offspring = np.zeros((self.npop, n * n))
    
    # Initialize the population
  
  def solve(self):
    pass