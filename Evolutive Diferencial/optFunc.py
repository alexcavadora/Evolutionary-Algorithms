import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from lineSearch import linSearch, BFGSOptimizer
from scipy.optimize import minimize

class tetaFunc():
  def __init__(self, l1 ,l2, Px, Py):
    self.l1 = l1
    self.l2 = l2
    self.Px = Px
    self.Py = Py

  def eval(self, teta):
    # teta = np.radians(teta)
    e1 = self.Px - self.l1*np.cos(teta[0]) - self.l2*np.cos(teta[1]) 
    e2 = self.Py - self.l1*np.sin(teta[0]) - self.l2*np.sin(teta[1])
    return e1**2 + e2**2
  
  def grad(self, teta):
    # teta = np.radians(teta)
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
    x = 0
    y = 0
    for t in self.t_values:
      x = x + self.x_amp * np.cos(self.w1 * t)
      y = y + self.y_amp * np.sin(self.w2 * t)
      self.x_traj.append(x)
      self.y_traj.append(y)
  
  def get_Xpos(self):
    return self.x0 + self.l1 * np.cos(self.teta1) + self.l2 * np.cos(self.teta2)

  def get_Ypos(self):
    return self.y0 + self.l1 * np.sin(self.teta1) + self.l2 * np.sin(self.teta2)

  def apply_PID(self,Px, Py):
    func = tetaFunc(self.l1, self.l2, Px, Py)
    # optimizer = linSearch(func, stop_criterion, step_condition)

    optimizer = minimize(func.eval, [self.teta1, self.teta2], jac=func.grad, method='BFGS')
    x_opt = optimizer.x

    self.teta1 = x_opt[0]
    self.teta2 = x_opt[1]
  

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

        # self.apply_PID(Px_new, Py_new)
        self.apply_PID(x_target, y_target)

    return sum_sq_error
      
  def plot_trajectory(self):
      
    plt.figure(figsize=(8, 6))
    plt.plot(self.x_traj, self.y_traj, label='Trajectory')
    plt.scatter(self.x0, self.y0, label='Inicio', color='red')
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
    plt.scatter(x_values[0], y_values[0], label='Inicio', color='red')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Position of the Particle')
    plt.legend()
    plt.grid(True)
    plt.show()

  def plot_both(self, teta_init = [45,45]):

    plt.figure(figsize=(8, 6))
    plt.plot(self.x_traj, self.y_traj, label='Trajectory', linestyle='--', alpha=0.75)
    # plt.plot  
    plt.scatter(self.x0, self.y0, label='punto fijo', color='red')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Trajectory of the Particle')
    plt.legend()
    plt.grid(True)

    
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
    
    plt.plot(x_values, y_values, label='Position')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Position of the Particle')
    plt.legend()
    plt.grid(True)
    plt.show()

  def plot_arm_mov(self, teta_init= [45,45]):
      import matplotlib.animation as animation
      
      # Convert initial angles to radians
      self.teta1, self.teta2 = np.radians(teta_init)
      
      # Initialize arrays to store positions
      x_end_values = []
      y_end_values = []
      x_joint_values = []
      y_joint_values = []
      
      # Reset PID controller variables
      x_prev_error = 0
      y_prev_error = 0
      x_integral = 0
      y_integral = 0
      
      # Calculate positions at each time step
      for t_idx in range(self.t_end):
        # Current end effector position
        Px = self.get_Xpos()
        Py = self.get_Ypos()
        # Joint position (where the two links connect)
        x_joint = self.l1 * np.cos(self.teta1)
        y_joint = self.l1 * np.sin(self.teta1)
        
        # Store positions
        x_end_values.append(Px)
        y_end_values.append(Py)
        x_joint_values.append(x_joint)
        y_joint_values.append(y_joint)
        
        # Target positions
        x_target = self.x_traj[t_idx]
        y_target = self.y_traj[t_idx]
        
        # PID control calculation
        x_error = x_target - Px
        y_error = y_target - Py
        
        x_proportional = self.kc * x_error
        x_integral += self.ki * x_error * self.delta_t
        x_derivative = self.kd * (x_error - x_prev_error) / self.delta_t
        u1 = x_proportional + x_integral + x_derivative
        
        y_proportional = self.kc * y_error
        y_integral += self.ki * y_error * self.delta_t
        y_derivative = self.kd * (y_error - y_prev_error) / self.delta_t
        u2 = y_proportional + y_integral + y_derivative
        
        x_prev_error, y_prev_error = x_error, y_error
        
        Px_new = Px + u1
        Py_new = Py + u2
        self.apply_PID(Px_new, Py_new)
      
      # Create figure for animation
      fig, ax = plt.subplots(figsize=(10, 8))
      
      # Set the limits of the plot
      max_reach = self.l1 + self.l2 + max(self.x_amp, self.y_amp) + 1
      ax.set_xlim(self.x0 - max_reach, self.x0 + max_reach)
      ax.set_ylim(self.y0 - max_reach, self.y0 + max_reach)
      
      # Plot the target trajectory
      ax.plot(self.x_traj, self.y_traj, 'b--', label='Target Trajectory')
      
      # Initialize arm segments
      line_segment1, = ax.plot([], [], 'r-', linewidth=3, label='Link 1')
      line_segment2, = ax.plot([], [], 'g-', linewidth=3, label='Link 2')
      joint_point, = ax.plot([], [], 'ko', markersize=8)
      end_point, = ax.plot([], [], 'bo', markersize=8)
      
      # Add origin point
      ax.plot(self.x0, self.y0, 'ro', markersize=10, label='Origin')
      
      ax.set_xlabel('X Position')
      ax.set_ylabel('Y Position')
      ax.set_title('Robotic Arm Animation')
      ax.legend()
      ax.grid(True)
      
      def init():
        line_segment1.set_data([], [])
        line_segment2.set_data([], [])
        joint_point.set_data([], [])
        end_point.set_data([], [])
        return line_segment1, line_segment2, joint_point, end_point
      
      def animate(i):
        # First segment: from origin to joint
        x1 = [self.x0, x_joint_values[i]]
        y1 = [self.y0, y_joint_values[i]]
        line_segment1.set_data(x1, y1)
        
        # Second segment: from joint to end effector
        x2 = [x_joint_values[i], x_end_values[i]]
        y2 = [y_joint_values[i], y_end_values[i]]
        line_segment2.set_data(x2, y2)
        
        joint_point.set_data([x_joint_values[i]], [y_joint_values[i]])
        end_point.set_data([x_end_values[i]], [y_end_values[i]])
        
        return line_segment1, line_segment2, joint_point, end_point
      
      anim = animation.FuncAnimation(fig, animate, init_func=init, 
                       frames=self.t_end, interval=self.delta_t*1000, 
                       blit=True)
      
      plt.show()
      
      # Uncomment to save animation
      anim.save('arm_animation.gif', writer='pillow', fps=int(1/self.delta_t))


    
if __name__ == "__main__":

  # Define the parameters for the PIDFunction
  x0 = [2.5, 5]  # Initial position
  t_end = 30  # End time for the simulation
  del_t = 0.25  # Time step for the simulation
  x = [5, 5]  # Amplitude of the trajectory in X and Y
  w = [1, 3]  # Frequency of the trajectory in X and Y
  teta = [45, 45]  #Initial Angles for the arm
  sides = [11, 11]  # Lengths of the arm sides

  # Create an instance of the PIDFunction
  pid_function = PIDFunction(x0, t_end,del_t, x, w, teta, sides)

  # Plot the trajectory
  # pid_function.plot_trajectory()

  # Evaluate the function with some parameters
  # 20.00, KI1 = 10.0000, KD1 = 0.0010
  k = [-6463.73281787, 44008.637523, 578.31744126]
  print(f'Error: {pid_function.evaluate(k= k)}')

  # Plot the position
  # pid_function.plot_position()

  pid_function.plot_both()
  # pid_function.plot_arm_mov()


    

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
  
