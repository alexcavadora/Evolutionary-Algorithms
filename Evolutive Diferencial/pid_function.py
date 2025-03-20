import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

class PIDFunction:
    def __init__(self, x0, t_end, delta_t=0.1, x=[1, 1], w=[1, 2], teta=[45, 45], sides=[1, 1]):
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
        self.teta1 = np.deg2rad(teta[0])
        self.teta2 = np.deg2rad(teta[1])
        self.l1 = sides[0]
        self.l2 = sides[1]

        self.Q1_history = []
        self.Q2_history = []
        self.integral_Q1 = 0.0
        self.integral_Q2 = 0.0
        self.previous_Q1 = 0.0
        self.previous_Q2 = 0.0

    def get_Xtrayectory(self, t):
        return self.x0 + self.x * np.cos(self.w1 * t)

    def get_Ytrayectory(self, t):
        return self.y0 + self.y * np.sin(self.w2 * t)

    def get_Xpos(self, t):
        # Correct forward kinematics for X position
        return self.l1 * np.cos(self.teta1) + self.l2 * np.cos(self.teta1 + self.teta2)

    def get_Ypos(self, t):
        # Correct forward kinematics for Y position
        return self.l1 * np.sin(self.teta1) + self.l2 * np.sin(self.teta1 + self.teta2)

    def get_Xerror(self, t):
        x_tray = self.get_Xtrayectory(t)
        x_pos = self.get_Xpos(t)
        return x_tray - x_pos

    def get_Yerror(self, t):
        y_tray = self.get_Ytrayectory(t)
        y_pos = self.get_Ypos(t)
        return y_tray - y_pos

    def get_Xpid(self, t):
        proportional = self.kc * self.get_Xerror(t)
        integral = self.ki * (integrate.quad(self.get_Xerror, 0, t)[0])
        delta = 1e-10
        derivative = self.kd * (self.get_Xerror(t) - self.get_Xerror(t - delta)) / delta
        return proportional + integral + derivative

    def get_Ypid(self, t):
        proportional = self.kc * self.get_Yerror(t)
        integral = self.ki * (integrate.quad(self.get_Yerror, 0, t)[0])
        delta = 1e-10
        derivative = self.kd * (self.get_Yerror(t) - self.get_Yerror(t - delta)) / delta
        return proportional + integral + derivative

    def apply_PID(self, t):
        # Apply PID control to both joint angles
        self.teta1 += self.get_Xpid(t) * self.delta_t
        self.teta2 += self.get_Ypid(t) * self.delta_t

    def evaluate(self, k):
        self.kc = k[0]
        self.ki = k[1]
        self.kd = k[2]
        sum_error = 0
        # Reset angles to initial values
        self.teta1 = np.deg2rad(45)
        self.teta2 = np.deg2rad(45)
        # Evaluate over the entire time range
        for t in np.arange(0, self.t_end, self.delta_t):
            self.apply_PID(t)
            sum_error += self.get_Xerror(t) ** 2 + self.get_Yerror(t) ** 2
        return sum_error

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
        arm_positions = []
        
        # Reset angles to initial values
        self.teta1 = np.deg2rad(45)
        self.teta2 = np.deg2rad(45)
        
        # Simulate the arm movement over time
        for t in t_values:
            self.apply_PID(t)
            x_pos = self.get_Xpos(t)
            y_pos = self.get_Ypos(t)
            x_values.append(x_pos)
            y_values.append(y_pos)
            
            # Store arm joint positions for visualization
            x0, y0 = 0, 0
            x1 = self.l1 * np.cos(self.teta1)
            y1 = self.l1 * np.sin(self.teta1)
            x2 = x1 + self.l2 * np.cos(self.teta1 + self.teta2)
            y2 = y1 + self.l2 * np.sin(self.teta1 + self.teta2)
            arm_positions.append(((x0, y0), (x1, y1), (x2, y2)))
        
        # Create the plot
        plt.figure(figsize=(8, 6))
        
        # Plot the arm's end effector path
        plt.plot(x_values, y_values, 'b-', label='Position')
        
        # Plot the desired trajectory
        t_values = np.arange(0, self.t_end, self.delta_t)
        x_traj = [self.get_Xtrayectory(t) for t in t_values]
        y_traj = [self.get_Ytrayectory(t) for t in t_values]
        plt.plot(x_traj, y_traj, 'orange', label='Trajectory')
        
        # Plot the final arm position
        last_pos = arm_positions[-1]
        plt.plot([last_pos[0][0], last_pos[1][0], last_pos[2][0]], 
                 [last_pos[0][1], last_pos[1][1], last_pos[2][1]], 
                 'r-o', label='Arm Position')
        
        # Plot the target position
        plt.plot(self.get_Xtrayectory(t_values[-1]), 
                 self.get_Ytrayectory(t_values[-1]), 
                 'g*', label='Target Position')
        
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Position of the Particle')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()