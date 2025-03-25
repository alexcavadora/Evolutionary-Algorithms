import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class PIDController:
    def __init__(self, max_integral=100.0):
        self.integral = 0.0
        self.prev_error = 0.0
        self.max_integral = max_integral
    
    def compute(self, e, kp, ki, kd, dt):
        self.integral = np.clip(self.integral + e*dt, -self.max_integral, self.max_integral)
        derivative = (e - self.prev_error)/dt if dt > 0 else 0.0
        self.prev_error = e
        return kp*e + ki*self.integral + kd*derivative

class PIDFunction:
    def __init__(self, x0, t_end=10, delta_t=0.01, l1=1.0, l2=1.0):
        self.l1, self.l2 = l1, l2
        self.x0, self.y0 = x0
        self.dt = delta_t
        self.t_end = t_end
        self.t_values = np.arange(0, t_end) * delta_t
        
        # Precalcular trayectoria deseada (Lemniscata)
        # self.x_traj = 1.0 * np.sin(1.0 * self.t_values)
        # self.y_traj = 1.0 * np.sin(1.0 * self.t_values) * np.cos(1.0 * self.t_values)
        
        # Precalcular trayectorias
        self.x_traj = []
        self.y_traj = []

        for t in self.t_values:
            x = 1 * np.cos(1* t)
            y = 1 * np.sin(2 * t)
            self.x_traj.append(x)
            self.y_traj.append(y)

        self.theta1, self.theta2 = np.radians([45, 45])
        self.k_params = None

    def forward_kinematics(self):
        return (
            self.x0 + self.l1*np.cos(self.theta1) + self.l2*np.cos(self.theta1 + self.theta2),
            self.y0 + self.l1*np.sin(self.theta1) + self.l2*np.sin(self.theta1 + self.theta2)
        )

    def compute_jacobian(self):
        s1 = np.sin(self.theta1)
        s12 = np.sin(self.theta1 + self.theta2)
        c1 = np.cos(self.theta1)
        c12 = np.cos(self.theta1 + self.theta2)
        return np.array([
            [-self.l1*s1 - self.l2*s12, -self.l2*s12],
            [self.l1*c1 + self.l2*c12,  self.l2*c12]
        ])

    def apply_control(self, u_x, u_y):
        J = self.compute_jacobian()
        try:
            delta_theta = np.linalg.pinv(J) @ np.array([u_x, u_y]) * self.dt
        except np.linalg.LinAlgError:
            delta_theta = np.zeros(2)
        self.theta1 += delta_theta[0]
        self.theta2 += delta_theta[1]

    def evaluate(self, k_params):
        self.k_params = k_params
        # kp1, ki1, kd1, kp2, ki2, kd2 = k_params
        kp, ki, kd = k_params
        pid_x = PIDController()
        pid_y = PIDController()
        total_error = 0.0
        
        for t_idx in range(len(self.t_values)):
            Px, Py = self.forward_kinematics()
            x_target, y_target = self.x_traj[t_idx], self.y_traj[t_idx]
            
            # Cálculo de errores y control PID
            e_x = x_target - Px
            e_y = y_target - Py
            # u_x = pid_x.compute(e_x, kp1, ki1, kd1, self.dt)
            # u_y = pid_y.compute(e_y, kp2, ki2, kd2, self.dt)
            u_x = pid_x.compute(e_x, kp, ki, kd, self.dt)
            u_y = pid_y.compute(e_y, kp, ki, kd, self.dt)

            self.apply_control(u_x, u_y)
            total_error += e_x**2 + e_y**2
            
        return total_error

    # def _simulate_trajectory(self):
    #     theta_history = []
    #     Px, Py = self.forward_kinematics()
    #     theta_history.append((Px, Py))
        
    #     for t_idx in range(len(self.t_values)):
    #         Px, Py = self.forward_kinematics()
    #         x_target, y_target = self.x_traj[t_idx], self.y_traj[t_idx]
            
    #         e_x = x_target - Px
    #         e_y = y_target - Py
    #         u_x = PIDController().compute(e_x, *self.k_params, self.dt)
    #         u_y = PIDController().compute(e_y, *self.k_params, self.dt)
            
    #         self.apply_control(u_x, u_y)
    #         theta_history.append(self.forward_kinematics())
            
    #     return np.array(theta_history)


    # def plot_arm_animation(self):
    #     trajectory = self._simulate_trajectory()
    #     fig, ax = plt.subplots(figsize=(8, 8))
    #     ax.set_xlim(-2, 2)
    #     ax.set_ylim(-2, 2)
        
    #     line, = ax.plot([], [], 'o-', lw=2)
    #     target_line, = ax.plot(self.x_traj, self.y_traj, 'r--')
        
    #     def animate(i):
    #         x_joint = self.x0 + self.l1 * np.cos(self.theta1)
    #         y_joint = self.y0 + self.l1 * np.sin(self.theta1)
    #         x_end, y_end = trajectory[i]
    #         line.set_data([self.x0, x_joint, x_end], 
    #                       [self.y0, y_joint, y_end])
    #         return line,
        
    #     ani = animation.FuncAnimation(fig, animate, frames=len(trajectory),
    #                                   interval=self.dt*1000, blit=True)
    #     plt.show()

    #     ani.save('arm_animation.gif', writer='pillow', fps=int(1/self.dt))
    
    def _simulate_trajectory(self):
        """Simula el movimiento del brazo y guarda el historial completo"""
        trajectory = []
        angles_history = []  # Para guardar los ángulos theta1 y theta2
        
        # Guardar valores iniciales
        Px, Py = self.forward_kinematics()
        trajectory.append((Px, Py))
        angles_history.append((self.theta1, self.theta2))
        
        # Restaurar ángulos iniciales para la simulación
        self.theta1, self.theta2 = np.radians([45, 45])
        
        for t_idx in range(len(self.t_values)):
            Px, Py = self.forward_kinematics()
            x_target, y_target = self.x_traj[t_idx], self.y_traj[t_idx]
            
            e_x = x_target - Px
            e_y = y_target - Py
            
            # Usar los parámetros PID optimizados
            kp, ki, kd = self.k_params
            u_x = PIDController().compute(e_x, kp, ki, kd, self.dt)
            u_y = PIDController().compute(e_y, kp, ki, kd, self.dt)
            
            self.apply_control(u_x, u_y)
            
            # Guardar posición y ángulos actuales
            trajectory.append(self.forward_kinematics())
            angles_history.append((self.theta1, self.theta2))
            
        return np.array(trajectory), angles_history

    def plot_comparison(self):
        trajectory,_ = self._simulate_trajectory()
        plt.figure(figsize=(10, 6))
        plt.plot(self.x_traj, self.y_traj, 'r--', label='Trayectoria deseada')
        plt.plot(trajectory[:,0], trajectory[:,1], 'b-', label='Trayectoria real')
        plt.scatter(self.x0, self.y0, c='g', s=100, label='Base')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_arm_animation(self):
        """Anima el movimiento del brazo robótico con la articulación intermedia"""
        trajectory, angles_history = self._simulate_trajectory()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        
        # Configurar elementos gráficos
        arm_line, = ax.plot([], [], 'o-', lw=2, markersize=6)
        target_point, = ax.plot([], [], 'ro', markersize=8)
        target_traj, = ax.plot(self.x_traj, self.y_traj, 'r--', alpha=0.5)
        actual_traj, = ax.plot([], [], 'b-', alpha=0.7)
        
        # Historial de posición real para trazar la trayectoria seguida
        traj_x, traj_y = [], []
    
        def init():
            arm_line.set_data([], [])
            target_point.set_data([], [])
            actual_traj.set_data([], [])
            return arm_line, target_point, actual_traj
        
        def animate(i):
            # Obtener los ángulos del historial
            theta1, theta2 = angles_history[i]
            
            # Calcular posiciones de las articulaciones con los ángulos históricos
            x_joint = self.x0 + self.l1 * np.cos(theta1)
            y_joint = self.y0 + self.l1 * np.sin(theta1)
            
            # Calcular posición del efector final
            x_end = self.x0 + self.l1 * np.cos(theta1) + self.l2 * np.cos(theta1 + theta2)
            y_end = self.y0 + self.l1 * np.sin(theta1) + self.l2 * np.sin(theta1 + theta2)
            
            # Dibujar el brazo robótico
            arm_line.set_data([self.x0, x_joint, x_end], [self.y0, y_joint, y_end])
            
            # Dibujar el punto objetivo
            if i < len(self.x_traj):
                target_point.set_data([self.x_traj[i]], [self.y_traj[i]])
            
            # Actualizar trayectoria real
            traj_x.append(x_end)
            traj_y.append(y_end)
            actual_traj.set_data(traj_x, traj_y)
            
            return arm_line, target_point, actual_traj
        
        ani = animation.FuncAnimation(fig, animate, frames=len(trajectory),
                                    init_func=init, interval=self.dt*1000, 
                                    blit=True)
        
        ax.grid(True)
        ax.set_title('Movimiento del Brazo Robótico')
        ax.legend([arm_line, target_traj, actual_traj], 
                ['Brazo', 'Trayectoria deseada', 'Trayectoria real'])
        
        plt.show()
        
        # Guardar la animación como GIF
        ani.save('arm_animation.gif', writer='pillow', fps=int(1/self.dt))

if __name__ == "__main__":
    pid_system = PIDFunction(x0=[0, 0], t_end=64, delta_t=0.0995, l1=0.75, l2=0.75)
    # best_params = [8.5, 0.2, 0.5, 8.5, 0.2, 0.5]  # Parámetros optimizados
    best_params = [8.5, 0.2, 0.5]  # Parámetros optimizados
    print("Error total:", pid_system.evaluate(best_params))
    pid_system.plot_comparison()
    pid_system.plot_arm_animation()