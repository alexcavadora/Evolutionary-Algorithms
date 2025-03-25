import numpy as np
import matplotlib.pyplot as plt

# =============================================
# 1. Configuración del sistema robótico
# =============================================
L1 = 1.0  # Longitud primer eslabón
L2 = 1.0  # Longitud segundo eslabón
dt = 0.01  # Paso de tiempo
t_max = 10.0  # Tiempo máximo de simulación
t_steps = np.arange(0, t_max, dt)  # Vector de tiempo

# =============================================
# 2. Cinemática directa y Jacobiano
# =============================================
def forward_kinematics(theta1, theta2):
    """Calcula la posición del efector final"""
    Px = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    Py = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
    return Px, Py

def compute_jacobian(theta1, theta2):
    """Calcula la matriz Jacobiana"""
    J11 = -L1 * np.sin(theta1) - L2 * np.sin(theta1 + theta2)
    J12 = -L2 * np.sin(theta1 + theta2)
    J21 = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    J22 = L2 * np.cos(theta1 + theta2)
    return np.array([[J11, J12], [J21, J22]])

# =============================================
# 3. Trayectoria deseada (Lemniscata de Bernoulli)
# =============================================
def desired_trajectory(t, a=1.0, omega=1.0):
    """Genera una trayectoria en forma de infinito"""
    Px_d = a * np.sin(omega * t)
    Py_d = a * np.sin(omega * t) * np.cos(omega * t)
    return Px_d, Py_d

# =============================================
# 4. Controlador PID Modificado
# =============================================

class PIDController:
    def __init__(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.max_integral = 100.0  # Límite anti-windup
    
    def compute(self, e, kp, ki, kd, dt):
        self.integral += e * dt
        self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)
        derivative = (e - self.prev_error) / dt if dt > 0 else 0.0
        output = kp * e + ki * self.integral + kd * derivative
        self.prev_error = e
        return output

# =============================================
# 5. Función de Costo Actualizada
# =============================================
def fitness_function(params):
    """Evalúa el desempeño de los parámetros PID con Jacobiano"""
    kp1, ki1, kd1, kp2, ki2, kd2 = params
    
    pid_x = PIDController()
    pid_y = PIDController()
    
    theta1, theta2 = 0.0, 0.0
    total_error = 0.0
    
    for t in t_steps:
        # Trayectoria deseada
        Px_d, Py_d = desired_trajectory(t)
        
        # Cinemática directa
        Px, Py = forward_kinematics(theta1, theta2)
        
        # Errores
        e_x = Px_d - Px
        e_y = Py_d - Py
        
        # Control PID
        u_x = pid_x.compute(e_x, kp1, ki1, kd1, dt)
        u_y = pid_y.compute(e_y, kp2, ki2, kd2, dt)
        
        # Calcular Jacobiano y actualizar ángulos
        J = compute_jacobian(theta1, theta2)
        try:
            # Usar pseudoinversa para evitar singularidades
            delta_theta = np.linalg.pinv(J) @ np.array([u_x, u_y]) * dt
        except np.linalg.LinAlgError:
            delta_theta = np.zeros(2)
        
        theta1 += delta_theta[0]
        theta2 += delta_theta[1]
        
        # Acumular error cuadrático
        total_error += e_x**2 + e_y**2
    
    return total_error

# =============================================
# 6. Algoritmo de Evolución Diferencial
# =============================================
def differential_evolution(bounds, pop_size=50, max_generations=100, F=0.8, CR=0.9):
    D = len(bounds)
    population = np.random.rand(pop_size, D)
    for i in range(D):
        low, high = bounds[i]
        population[:, i] = low + population[:, i] * (high - low)
    
    best_solution = None
    best_fitness = float('inf')
    
    for gen in range(max_generations):
        fitness = np.array([fitness_function(ind) for ind in population])
        
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx]
            best_solution = population[current_best_idx].copy()
        
        for i in range(pop_size):
            # Mutación DE/rand/1
            candidates = [idx for idx in range(pop_size) if idx != i]
            a, b, c = np.random.choice(candidates, 3, replace=False)
            mutant = population[a] + F * (population[b] - population[c])
            
            # Cruce binomial
            cross_points = np.random.rand(D) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(D)] = True
            trial = np.where(cross_points, mutant, population[i])
            
            # Aplicar límites
            trial = np.clip(trial, [b[0] for b in bounds], [b[1] for b in bounds])
            
            trial_fitness = fitness_function(trial)
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
        
        print(f"Generación {gen+1}: Mejor fitness = {best_fitness:.4f}")
    
    return best_solution, best_fitness

# =============================================
# 7. Configuración y ejecución
# =============================================
bounds = [
    (0.1, 20.0), (0.001, 10.0), (0.001, 10.0),  # PID para X
    (0.1, 20.0), (0.001, 10.0), (0.001, 10.0)   # PID para Y
]

best_params, best_error = differential_evolution(
    bounds=bounds,
    pop_size=60,
    max_generations=50,
    F=0.75,
    CR=0.85
)

# =============================================
# 8. Simulación final con mejores parámetros
# =============================================
def simulate_and_plot(best_params):
    kp1, ki1, kd1, kp2, ki2, kd2 = best_params
    
    pid_x = PIDController()
    pid_y = PIDController()
    
    theta1, theta2 = 0.0, 0.0
    history = {'t': [], 'Px': [], 'Py': [], 'Px_d': [], 'Py_d': [], 'theta1': [], 'theta2': []}
    
    for t in t_steps:
        Px_d, Py_d = desired_trajectory(t)
        Px, Py = forward_kinematics(theta1, theta2)
        
        # Registrar datos
        history['t'].append(t)
        history['Px'].append(Px)
        history['Py'].append(Py)
        history['Px_d'].append(Px_d)
        history['Py_d'].append(Py_d)
        history['theta1'].append(theta1)
        history['theta2'].append(theta2)
        
        # Control PID
        e_x = Px_d - Px
        e_y = Py_d - Py
        u_x = pid_x.compute(e_x, kp1, ki1, kd1, dt)
        u_y = pid_y.compute(e_y, kp2, ki2, kd2, dt)
        
        # Actualización con Jacobiano
        J = compute_jacobian(theta1, theta2)
        try:
            delta_theta = np.linalg.pinv(J) @ np.array([u_x, u_y]) * dt
        except np.linalg.LinAlgError:
            delta_theta = np.zeros(2)
        
        theta1 += delta_theta[0]
        theta2 += delta_theta[1]
    
    # Gráficos
    plt.figure(figsize=(15, 8))
    
    # Trayectoria
    plt.subplot(2, 2, 1)
    plt.plot(history['Px_d'], history['Py_d'], 'r--', label='Deseada')
    plt.plot(history['Px'], history['Py'], 'b-', label='Real')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trayectoria del efector final')
    plt.legend()
    plt.grid(True)
    
    # Error
    plt.subplot(2, 2, 2)
    error = np.sqrt((np.array(history['Px_d']) - np.array(history['Px']))**2 + 
                    (np.array(history['Py_d']) - np.array(history['Py']))**2)
    plt.plot(history['t'], error)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Error (m)')
    plt.title('Evolución del error')
    plt.grid(True)
    
    # Ángulos articulares
    plt.subplot(2, 2, 3)
    plt.plot(history['t'], np.degrees(history['theta1']), label='Theta1')
    plt.plot(history['t'], np.degrees(history['theta2']), label='Theta2')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Ángulo (grados)')
    plt.title('Evolución de los ángulos articulares')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Ejecutar simulación final
print("\nMejores parámetros PID:")
print(f"KP1 = {best_params[0]:.2f}, KI1 = {best_params[1]:.4f}, KD1 = {best_params[2]:.4f}")
print(f"KP2 = {best_params[3]:.2f}, KI2 = {best_params[4]:.4f}, KD2 = {best_params[5]:.4f}")
simulate_and_plot(best_params)