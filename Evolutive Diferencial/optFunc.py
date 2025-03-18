import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from lineSearch import linSearch, BFGSOptimizer

class tetaFunc():
  def __init__(self, l1 ,l2, Px, Py):
    self.l1 = l1
    self.l2 = l2
    self.Px = Px
    self.Py = Py

  def eval(self, teta):
    e1 = self.Px - self.l1*np.cos(teta[0]) - self.l2*np.cos(teta[1]) 
    e2 = self.Py - self.l1*np.sin(teta[0]) - self.l2*np.sin(teta[1])
    return e1**2 + e2**2
  
  def grad(self, teta):
    e1 = self.Px - self.l1*np.cos(teta[0]) - self.l2*np.cos(teta[1]) 
    e2 = self.Py - self.l1*np.sin(teta[0]) - self.l2*np.sin(teta[1])
    grad1 = 2*self.l1*( e1*np.sin(teta[0]) - e2*np.cos(teta[0]) )
    grad2 = 2*self.l2*( e1*np.sin(teta[1]) - e2*np.cos(teta[1]) )
    gradient = np.array([grad1, grad2])
    return gradient

class PIDFunction():
  def __init__(self, x0, t_end, delta_t=0.1, x=[1,1], w=[1,2], teta=[45,45], sides=[1,1]):
    self.kc = None
    self.ki = None
    self.kd = None
    self.t_end = t_end
    self.delta_t = delta_t
    self.x0 = x0[0]  # Posición inicial X
    self.y0 = x0[1]  # Posición inicial Y
    self.x_amp = x[0]  # Amplitud en X
    self.y_amp = x[1]  # Amplitud en Y
    self.w1 = w[0]
    self.w2 = w[1]
    self.teta1 = np.radians(teta[0])
    self.teta2 = np.radians(teta[1])  # Ángulo en grados
    self.l1 = sides[0]
    self.l2 = sides[1]
    # Precalcular trayectorias
    self.x_traj = []
    self.y_traj = []
    self.t_values = np.arange(0, self.t_end) * self.delta_t
    x = self.get_Xpos()
    y = self.get_Ypos()
    for t in self.t_values:
      x = x + self.x_amp * np.cos(self.w1 * t)
      y = y + self.y_amp * np.sin(self.w2 * t)
      self.x_traj.append(x)
      self.y_traj.append(y)
  
  def get_Xpos(self):
    return self.l1 * np.cos(self.teta1) + self.l2 * np.cos(np.radians(self.teta2))

  def get_Ypos(self):
    return self.l1 * np.sin(self.teta1) + self.l2 * np.sin(np.radians(self.teta2))

  def apply_PID(self,Px, Py):
    func = tetaFunc(self.l1, self.l2, Px, Py)
    # optimizer = linSearch(func, stop_criterion, step_condition)
    optimizer = BFGSOptimizer(func)
    teta_i = [self.teta1, self.teta2]
    # teta_i = np.random.rand(2)
    x_opt = optimizer.optimize(teta_i)
    self.teta1 = x_opt[0]
    self.teta2 = x_opt[1]
  
  # def apply_PID(self, Px, Py):
  #   # Cinemática inversa analítica
  #   l1, l2 = self.l1, self.l2
  #   x, y = Px, Py
  #   d_sq = x**2 + y**2
  #   d = np.sqrt(d_sq)
    
  #   # Si el punto está fuera de alcance, lo ajustamos al límite alcanzable
  #   if d > l1 + l2:
  #       # Normalizar y escalar al máximo alcance
  #       factor = (l1 + l2) / d
  #       x *= factor
  #       y *= factor
  #       d = l1 + l2
  #   elif d < abs(l1 - l2):
  #       # Ajustar para el alcance mínimo
  #       if d == 0:
  #           # Manejar caso especial en origen
  #           x = abs(l1 - l2)
  #           y = 0
  #       else:
  #           factor = abs(l1 - l2) / d
  #           x *= factor
  #           y *= factor
  #       d = abs(l1 - l2)
    
  #   # Cálculo de theta2
  #   cos_theta2 = (d_sq - l1**2 - l2**2) / (2 * l1 * l2)
  #   cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
  #   # Elegir solución de "codo arriba"
  #   theta2 = np.arccos(cos_theta2)
    
  #   # Cálculo de theta1
  #   gamma = np.arctan2(y, x)
  #   beta = np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))
  #   theta1 = gamma - beta
    
  #   # Actualizar ángulos en grados
  #   self.teta1 = np.degrees(theta1)
  #   self.teta2 = np.degrees(theta2)

  def evaluate(self, k, teta_init=None):
    self.kc, self.ki, self.kd = k
    if teta_init is None:
        teta_init = [np.radians(45), np.radians(45)]  # Default en radianes
    else:
      self.teta1, self.teta2 = np.radians(teta_init)

    sum_sq_error = 0
    x_prev_error = 0
    y_prev_error = 0
    x_integral = 0
    y_integral = 0

    for t_idx in range(self.t_end):
        Px = self.get_Xpos()
        Py = self.get_Ypos()
        x_target = self.x_traj[t_idx]
        y_target = self.y_traj[t_idx]

        # Errores
        x_error = x_target - Px
        y_error = y_target - Py
        sum_sq_error += x_error**2 + y_error**2

        # PID para X
        x_proportional = self.kc * x_error
        x_integral += self.ki * x_error * self.delta_t
        x_derivative = self.kd * (x_error - x_prev_error) / self.delta_t
        u1 = x_proportional + x_integral + x_derivative

        # PID para Y
        y_proportional = self.kc * y_error
        y_integral += self.ki * y_error * self.delta_t
        y_derivative = self.kd * (y_error - y_prev_error) / self.delta_t
        u2 = y_proportional + y_integral + y_derivative

        # Actualizar errores previos
        x_prev_error, y_prev_error = x_error, y_error

        # Aplicar control y cinemática inversa
        Px_new = Px + u1
        Py_new = Py + u2
        self.apply_PID(Px_new, Py_new)

    return sum_sq_error
      
  def plot_trajectory(self):
      
    plt.figure(figsize=(8, 6))
    plt.plot(self.x_traj, self.y_traj, label='Trajectory')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Trajectory of the Particle')
    plt.legend()
    plt.grid(True)
    plt.show()

  def plot_position(self, teta_init = [45,45]):

    x_values = []
    y_values = []
    self.teta1, self.teta2 = teta_init
    x_prev_error = 0
    y_prev_error = 0
    x_integral = 0
    y_integral = 0

    for t_idx in range(self.t_end):
        Px = self.get_Xpos()
        Py = self.get_Ypos()
        x_values.append(Px)
        y_values.append(Py)
        x_target = self.x_traj[t_idx]
        y_target = self.y_traj[t_idx]

        # Errores
        x_error = x_target - Px
        y_error = y_target - Py

        # PID para X
        x_proportional = self.kc * x_error
        x_integral += self.ki * x_error * self.delta_t
        x_derivative = self.kd * (x_error - x_prev_error) / self.delta_t
        u1 = x_proportional + x_integral + x_derivative

        # PID para Y
        y_proportional = self.kc * y_error
        y_integral += self.ki * y_error * self.delta_t
        y_derivative = self.kd * (y_error - y_prev_error) / self.delta_t
        u2 = y_proportional + y_integral + y_derivative

        # Actualizar errores previos
        x_prev_error, y_prev_error = x_error, y_error

        # Aplicar control y cinemática inversa
        Px_new = Px + u1
        Py_new = Py + u2
        self.apply_PID(Px_new, Py_new)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, label='Position')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Position of the Particle')
    plt.legend()
    plt.grid(True)
    plt.show()


    
if __name__ == "__main__":
  # Example usage:
  # Define the parameters for the PIDFunction
  x0 = [0, 0]  # Initial position
  t_end = 30  # End time for the simulation
  del_t = 0.25  # Time step for the simulation
  x = [10, 10]  # Amplitude of the trajectory in X and Y
  w = [1, 3]  # Frequency of the trajectory in X and Y
  teta = [45, 45]  # Angles for the arm
  sides = [3, 3]  # Lengths of the arm sides

  # Create an instance of the PIDFunction
  pid_function = PIDFunction(x0, t_end,del_t, x, w, teta, sides)

  # Plot the trajectory
  pid_function.plot_trajectory()

  # Evaluate the function with some parameters
  print(f'Error: {pid_function.evaluate(k= [10, 0.1, 0.01])}')

  # Plot the position
  pid_function.plot_position()


    

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
  

# if __name__ == "__main__":
#   # Example with a 5D vector
#   zero = 1e-16
#   x = [zero for i in range(5)]
#   print(f"Initial x: {x}")
#   # Initialize the Ackley function
#   ackley = AckleyFunction(x)

  
#   # Evaluate function at x
#   value = ackley.eval(x)
#   print(f"Function value f(x) = {value}")
#   print("-" * 40)

  
#   # Calculate gradient at x
#   gradient = ackley.grad(x)
#   print("Gradient at x:")
#   print(gradient)
#   print("-" * 40)
  
#   # Calculate Hessian at x
#   hessian = ackley.hess(x)
#   print("Hessian matrix at x:")
#   print(hessian)
#   print("-" * 40)
