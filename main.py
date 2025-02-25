import numpy as np

def generateTableOfBytes(n):
  return bytearray(n*n)

def prettyChessTablePrint(chess_table, table_size, printBinary=False):
  for row in range(0, len(chess_table), table_size):
    for col in range(table_size):
      pos = row + col
      if chess_table[pos] == 1:
        if printBinary:
          print(' 1 ', end='')
        else:
          print(' Q ', end='')
      elif chess_table[pos] == 0:
        if printBinary:
          print(' 0 ', end='')
        else:
          print(' . ', end='')
    print()
  print()
    
def defaultFuncInit(pop, table_size):
  for individual in pop:
    available_cols = list(range(table_size))
    # Generate a queen in each row randomly
    for row in range(0, len(individual), table_size):
      # Avoid generating a queen in the same column
      col = np.random.choice(available_cols)
      available_cols.remove(col)
      individual[row + col] = 1
  return pop

def defaultFEval(pop):
  fitness = np.zeros(len(pop))

  # Detect all the collisions using a mask
  

class GA:
  '''
    table_size: Size of the chess table
    nPop: Number of individuals in the population
    fEval: Fitness function
    nGen: Number of generations
    pMut: Probability of mutation
    pCross: Probability of crossover
    ptrFuncInit: Function to initialize the population
  '''
  def __init__(self, table_size=8, nPop=100, fEval=None, nGen=100, pMut=0.1, pCross=0.5, ptrFuncInit=defaultFuncInit):
    self.table_size = table_size
    self.nPop = nPop
    self.fEval = fEval
    self.nGen = nGen
    self.pMut = pMut
    self.pCross = pCross
    self.ptrFuncInit = ptrFuncInit

    self.pop = [generateTableOfBytes(table_size) for _ in range(nPop)]
    self.pop_fit = np.zeros(nPop)
    self.pop_select = self.pop
    self.offspring = self.pop
    
    # Initialize the population
    self.pop = self.ptrFuncInit(self.pop, table_size)
  def solve(self):
    pass

ga = GA()