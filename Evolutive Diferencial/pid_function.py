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
        return self.l1 * np.cos(self.teta1) + self.l2 * np.cos(self.teta2)

    def get_Ypos(self, t):
        return self.l1 * np.sin(self.teta1) + self.l2 * np.sin(self.teta2)

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
        self.teta1 += self.get_Xpid(t) * self.delta_t
        self.teta2 += self.get_Ypid(t) * self.delta_t

    def evaluate(self, k):
        self.kc = k[0]
        self.ki = k[1]
        self.kd = k[2]
        sum = 0
        self.teta1 = np.deg2rad(45)
        self.teta2 = np.deg2rad(45)
        for t in np.arange(0, self.t_end, self.delta_t):
            self.apply_PID(t)
            sum += self.get_Xerror(t) ** 2 + self.get_Yerror(t) ** 2
        return sum

    def plot_trajectory(self):
        t_values = np.arange(0, self.t_end, self.delta_t)
        x_values = [self.get_Xtrayectory(t) for t in t_values]
        y_values = [self.get_Ytrayectory(t) for t in t_values]
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
        self.teta1 = np.deg2rad(45)
        self.teta2 = np.deg2rad(45)
        for t in t_values:
            self.apply_PID(t)
            x_values.append(self.get_Xpos(t))
            y_values.append(self.get_Ypos(t))
        plt.figure(figsize=(8, 6))
        plt.plot(x_values, y_values, label='Position')

        t_values = np.arange(0, self.t_end, self.delta_t)
        x_values = [self.get_Xtrayectory(t) for t in t_values]
        y_values = [self.get_Ytrayectory(t) for t in t_values]
        plt.plot(x_values, y_values, label='Trajectory')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Trajectory of the Particle')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Add arm visualization at last position
        x0, y0 = 0, 0
        x1 = self.l1 * np.cos(self.teta1)
        y1 = self.l1 * np.sin(self.teta1)
        x2 = x1 + self.l2 * np.cos(self.teta2)
        y2 = y1 + self.l2 * np.sin(self.teta2)
        
        plt.plot([x0, x1, x2], [y0, y1, y2], 'r-o', label='Arm Position')
        plt.plot(self.get_Xtrayectory(t), self.get_Ytrayectory(t), 'g*', label='Target Position')
        
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Position of the Particle')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()