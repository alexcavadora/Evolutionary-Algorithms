import numpy as np

def generateTableOfBits(n):
  bytes_needed = int(np.ceil((n*n) / 8))
  return bytearray(bytes_needed)

def prettyChessTablePrint(chess_table, table_size, printBinary=False):
  for i in range(table_size):
    row = bin(chess_table[i]) # Convert to binary
    row = row.replace('0b', '') # Delete the '0b' prefix
    row = row.rjust(table_size, '0') # Pad with zeros to the right


    # print(bin(chess_table[row]).replace('0b', ''))
  # binary = bin(chess_table).replace('0b', '')
  # print(binary)
  # for row in range(0, table_size**2, table_size):
  #   for col in range(table_size):
  #     pos = row + col
  #     if chess_table[pos] == 1:
  #       if printBinary:
  #         print(' 1 ', end='')
  #       else:
  #         print(' Q ', end='')
  #     elif chess_table[pos] == 0:
  #       if printBinary:
  #         print(' 0 ', end='')
  #       else:
  #         print(' . ', end='')
  #   print()
  # print()
    
def defaultFuncInit(pop, table_size):
  pass
#   for individual in pop:
#     available_cols = list(range(table_size))
#     # Generate a queen in each row randomly
#     for row in range(0, len(individual), table_size):
#       # Avoid generating a queen in the same column
#       col = np.random.choice(available_cols)
#       available_cols.remove(col)
#       individual[row + col] = 1
#   return pop

# def defaultFEval(pop):
#   fitness = np.zeros(len(pop))

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

    self.pop = [generateTableOfBits(table_size) for _ in range(nPop)]
    self.pop_fit = np.zeros(nPop)
    self.pop_select = self.pop
    self.offspring = self.pop

    prettyChessTablePrint(self.pop[0], self.table_size)
    
    # Initialize the population
    # self.pop = self.ptrFuncInit(self.pop, table_size)
  def solve(self):
    pass

ga = GA()