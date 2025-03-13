import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

class PIDFunction():
  def __init__(self,x0,t_end,delta_t=0.1,x=[1,1],w=[1,2],teta=[45,45],sides = [1,1]):
    self.kc = None
    self.ki = None
    self.kd = None
    self.t_end = t_end
    self.delta_t = delta_t
    self.x0 = x0[0]
    self.y0 = x0[1]
    self.x = x[0]
    self.y = x[1]
    self.w1 = w[0]
    self.w2 = w[1]
    self.teta1 = teta[0]
    self.teta2 = teta[1]
    self.l1 = sides[0]
    self.l2 = sides[1]

    # Historial de errores para términos integral y derivativo
    self.Q1_history = []
    self.Q2_history = []
    self.integral_Q1 = 0.0
    self.integral_Q2 = 0.0
    self.previous_Q1 = 0.0
    self.previous_Q2 = 0.0


  def get_Xtrayectory(self, t):
    return self.x0 + self.x*np.cos(self.w1*t)
  
  def get_Ytrayectory(self, t):
    return self.y0 + self.y*np.sin(self.w2*t)
  
  def get_Xpos(self, t):
    # return self.l1*np.cos(self.teta1) + self.l2*np.cos(self.teta1 + self.teta2)
    return self.l1*np.cos(self.teta1) + self.l2*np.cos(self.teta2)

  def get_Ypos(self, t):
    # return self.l1*np.sin(self.teta1) + self.l2*np.sin(self.teta1 + self.teta2)
    return self.l1*np.sin(self.teta1) + self.l2*np.sin(self.teta2)
  
  def get_Xerror(self, t):
    x_tray = self.get_Xtrayectory(t)
    x_pos = self.get_Xpos(t)
    return x_tray - x_pos

  def get_Yerror(self, t):
    y_tray = self.get_Ytrayectory(t)
    y_pos = self.get_Ypos(t)
    return y_tray - y_pos

  def get_Xpid(self, t):
    proportional = self.kc*self.get_Xerror(t)
    integral = self.ki*(integrate.quad(self.get_Xerror, 0, t)[0])
    delta = 1e-10
    derivative = self.kd*(self.get_Xerror(t) - self.get_Xerror(t-delta))/delta
    return proportional + integral + derivative
  
  def get_Ypid(self, t):
    proportional = self.kc*self.get_Yerror(t)
    integral = self.ki*(integrate.quad(self.get_Yerror, 0, t)[0])
    delta = 1e-10
    derivative = self.kd*(self.get_Yerror(t) - self.get_Yerror(t-delta))/delta
    return proportional + integral + derivative

  def apply_PID(self, t):
    # self.teta1 += self.get_Xpid(t)
    # self.teta2 += self.get_Ypid(t)
    self.teta1 += self.get_Xpid(t)*self.delta_t
    self.teta2 += self.get_Ypid(t)*self.delta_t

  def evaluate(self, k):
    self.kc = k[0]
    self.ki = k[1]
    self.kd = k[2]
    sum = 0
    self.teta1 = 45
    self.teta2 = 45
    for t in range(self.t_end):
      self.apply_PID(t)
      sum += self.get_Xerror(t)**2 + self.get_Yerror(t)**2
    return sum
  
  def gradiente_fopt(theta1, theta2, L1, L2):
    # Derivada parcial con respecto a theta1
    df_dtheta1 = L1 * np.sin(theta1) - L1 * np.cos(theta1)
    
    # Derivada parcial con respecto a theta2
    df_dtheta2 = L2 * np.sin(theta2) - L2 * np.cos(theta2)
    
    # Retornar el gradiente como un array de numpy
    return np.array([df_dtheta1, df_dtheta2])

  def forward_kinematics(self):
    """Calcula la posición actual (x, y) usando los ángulos teta1 y teta2."""
    self.x = self.l1 * np.cos(self.teta1) + self.l2 * np.cos(self.teta2)
    self.y = self.l1 * np.sin(self.teta1) + self.l2 * np.sin(self.teta2)
    return self.x, self.y

  def solve(self, target_x, target_y, tol=1e-4):
      """
      Encuentra teta1 y teta2 óptimos usando un control PID.
      
      Parámetros:
      - target_x, target_y: Posición objetivo.
      - tol: Tolerancia para la convergencia.
      
      Retorna:
      - teta1, teta2: Ángulos óptimos en radianes.
      """
      # Verificar que las ganancias PID estén configuradas
      if self.kc is None or self.ki is None or self.kd is None:
          raise ValueError("Configura las ganancias kc, ki, kd primero.")

      # Número de pasos de tiempo
      n_steps = int(self.t_end / self.delta_t)
      
      for _ in range(n_steps):
          # Calcular posición actual
          current_x, current_y = self.forward_kinematics()
          
          # Calcular errores
          Q1 = target_x - current_x
          Q2 = target_y - current_y
          
          # Actualizar términos integrales
          self.integral_Q1 += Q1 * self.delta_t
          self.integral_Q2 += Q2 * self.delta_t
          
          # Calcular términos derivativos (usando diferencia finita)
          derivative_Q1 = (Q1 - self.previous_Q1) / self.delta_t if len(self.Q1_history) > 0 else 0.0
          derivative_Q2 = (Q2 - self.previous_Q2) / self.delta_t if len(self.Q2_history) > 0 else 0.0
          
          # Calcular acciones de control PID
          U1 = self.kc * Q1 + self.ki * self.integral_Q1 + self.kd * derivative_Q1
          U2 = self.kc * Q2 + self.ki * self.integral_Q2 + self.kd * derivative_Q2
          
          # Actualizar ángulos (usando velocidades angulares)
          self.teta1 += U1 * self.delta_t
          self.teta2 += U2 * self.delta_t
          
          # Guardar errores para el próximo paso
          self.previous_Q1 = Q1
          self.previous_Q2 = Q2
          
          # Verificar convergencia
          if np.sqrt(Q1**2 + Q2**2) < tol:
              break
      
      return self.teta1, self.teta2
      
  def plot_trajectory(self):

    t_values = np.arange(0, self.t_end, self.delta_t)
    x_values = [self.get_Xtrayectory(t) for t in t_values]
    y_values = [self.get_Ytrayectory(t) for t in t_values]
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, label='Trajectory')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Trajectory of the Particle')
    plt.legend()
    plt.grid(True)
    plt.show()

  def plot_position(self):
    t_values = np.arange(0, self.t_end, self.delta_t)
    x_values = []
    y_values = []
    self.teta1 = 45
    self.teta2 = 45
    for t in t_values:
      self.apply_PID(t)
      x_values.append(self.get_Xpos(t))
      y_values.append(self.get_Ypos(t))
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
  t_end = 7  # End time for the simulation
  del_t = 0.1  # Time step for the simulation
  x = [10, 10]  # Amplitude of the trajectory in X and Y
  w = [1, 2]  # Frequency of the trajectory in X and Y
  teta = [45, 45]  # Angles for the arm
  sides = [3, 3]  # Lengths of the arm sides  

  # Create an instance of the PIDFunction
  pid_function = PIDFunction(x0, t_end,del_t, x, w, teta, sides)

  # Plot the trajectory
  pid_function.plot_trajectory()

  # Evaluate the function with some parameters
  print(f'Error: {pid_function.evaluate([10, 1, 0.01])}')

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
