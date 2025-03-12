import numpy as np

class AckleyFunction():

  def __init__(self,x):
    self.d = len(x)
    self.a = 20
    self.b = 0.2
    self.c = 2 * np.pi
    self.e = np.ones(self.d)

  def eval(self, x):
    x_np = np.array(x).reshape(-1, 1)  # Convertir a vector columna
    return -self.a * np.exp(-self.b * np.sqrt(1.0/self.d * x_np.T @ x_np )) - np.exp(1.0/self.d * self.e.T @ np.cos(self.c * x_np) ) + self.a + np.exp(1)

  # def eval(self, x):
    # return -self.a * np.exp(-self.b * np.sqrt(1.0/self.d * x.T * x )) - np.exp(1.0/self.d * self.e.T * np.cos(self.c * x) ) + self.a + np.exp(1)

  def ident(self):
    n = self.d
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

  def grad(self, x):
    x_np = np.array(x).reshape(-1, 1)
    raiz = np.sqrt(1.0/self.d * x_np.T * x_np)
    uno = x_np *self.a *self.b/raiz * np.exp(-self.b * raiz)   
    dos = self.c * np.sin(self.c * x_np) * np.exp(1.0/self.d)
    tres = self.e.T * np.cos(self.c * x_np)                                           
                                              
    return (1.0/self.d)*(uno + dos * tres)

  def hess(self, x):
    x_np = np.array(x).reshape(-1, 1)
    raiz = np.sqrt(1.0/self.d * x_np.T * x_np)
    ex1 = np.exp(-self.b * raiz)
    ex2 = np.exp(1.0/self.d * self.e.T * np.cos(self.c * x_np))
    I = self.ident()
    cx = self.c * x_np
    senx = np.sin(cx)

    return (1.0/self.d)*( ( x_np * x_np.T * self.a/self.d + ( I + ( x_np * x_np.T )*2 ) * ex1 ) * (-self.b/raiz**2) + ( np.diag(np.cos(cx)) + senx * senx.T * 1.0/self.d) * self.c**2 * ex2)
  

if __name__ == "__main__":
  # Example with a 5D vector
  zero = 1e-16
  x = [zero for i in range(5)]
  print(f"Initial x: {x}")
  # Initialize the Ackley function
  ackley = AckleyFunction(x)

  
  # Evaluate function at x
  value = ackley.eval(x)
  print(f"Function value f(x) = {value}")
  print("-" * 40)

  
  # Calculate gradient at x
  gradient = ackley.grad(x)
  print("Gradient at x:")
  print(gradient)
  print("-" * 40)
  
  # Calculate Hessian at x
  hessian = ackley.hess(x)
  print("Hessian matrix at x:")
  print(hessian)
  print("-" * 40)
