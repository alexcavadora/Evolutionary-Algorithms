import numpy as np
import matplotlib.pyplot as plt
import time 

## necesito la funcion exit()
from sys import exit

class BFGSOptimizer:
    def __init__(self, func, c1=1e-4, c2=0.9, grad_tol=1e-4, max_iter=200):
        self.func = func
        self.c1 = c1          # Condición de Armijo
        self.grad_tol = grad_tol  # Tolerancia de gradiente
        self.max_iter = max_iter # Máximo de iteraciones

    def _armijo_condition(self, x, alpha, d):
        return self.func.eval(x + alpha * d) <= self.func.eval(x) + self.c1 * alpha * np.dot(self.func.grad(x).T, d)

    def _line_search(self, x, d):
        alpha = 1.0
        for _ in range(20):  # Límite más ajustado para backtracking
            if self._armijo_condition(x, alpha, d):
                return alpha
            alpha *= 0.5
        return alpha  # Devuelve el mejor alpha encontrado

    def optimize(self, x0):
        H = np.eye(len(x0))  # Inicialización de la Hessiana inversa
        grad = self.func.grad(x0)
        x = x0.copy()
        
        for _ in range(self.max_iter):
            # Criterio de parada
            if np.linalg.norm(grad) < self.grad_tol:
                break
            
            d = -H @ grad  # Dirección de descenso
            
            # Búsqueda de línea
            alpha = self._line_search(x, d)
            s = alpha * d
            x_next = x + s
            
            # Actualización de la Hessiana
            grad_next = self.func.grad(x_next)
            y = grad_next - grad
            rho = 1 / (y.T @ s + 1e-10)  # Evitar división por cero
            
            # Actualización BFGS en forma compacta
            H = (np.eye(len(x)) - rho * s @ y.T) @ H @ (np.eye(len(x)) - rho * y @ s.T) + rho * s @ s.T
            
            # Preparar siguiente iteración
            x = x_next
            grad = grad_next
        
        return x

class linSearch:
    def __init__(self, func, stopCrit, stepCond, c1=1e-6, c2=0.9):
        self.func = func
        self.stopCrit = stopCrit 
        self.stepCond = stepCond
        self.wolfe = WolfeConditions(c1, c2)
        self.x_history = []

    def GradientDescentMethod(self, x):
        return -(self.func.grad(x) / np.linalg.norm(self.func.grad(x)))

    def NewtonMethod(self, x):
        B_k =  self.func.hess(x)
        return -np.linalg.inv(B_k) @ self.func.grad(x)

    def optMethod(self, op, x):
        if op == 1:
            return self.NewtonMethod(x)
        elif op == 2:
            return self.GradientDescentMethod(x)
        else:
            raise ValueError("Elección no válida")

    def solve(self, x0, condition="armijo", method=2):
        """Resuelve el problema de optimización usando la condición de Wolfe especificada."""
        # x = np.array(x0, dtype=float)
        x = x0.copy()
        self.x_history.append(x.copy())
        iterations = 0

        while True:
            grad_norm = np.linalg.norm(self.func.grad(x))
            prev_value = self.x_history[-1] if len(self.x_history) > 1 else x
            curr_value = x
            
            if self.stopCrit(grad_norm, prev_value, curr_value, iterations):
                break

            d = self.optMethod(method, x)
            
            alpha = self.stepCond(x, d, self.func, self.wolfe, condition) ## error
            x = x + alpha * d
            self.x_history.append(x.copy())
            iterations += 1
            # print(f"Iteración {iterations}, grad_norm = {grad_norm}")
        self.x_history = np.array(self.x_history)
        return x

    def BFGS(self, x0, condition="armijo", op=1):
        self.x_history.append(x0.copy())
        p_k = self.optMethod(op, x0)
        alpha = self.stepCond(x0, p_k, self.func, self.wolfe, condition)
        xk = x0 + alpha * p_k
        H0 = np.eye(len(x0))  # Inicializar H0 como la matriz identidad
        I = np.eye(len(x0))   # Matriz identidad
        iterations = 0
        while True:
            grad_norm = np.linalg.norm(self.func.grad(x0))
            
            if self.stopCrit(grad_norm, iterations):
                break

            s_k = xk - x0  # Diferencia entre los puntos actual y anterior
            y_k = self.func.grad(xk) - self.func.grad(x0) # Diferencia entre gradientes
            rho_k = 1.0 / (y_k.T @ s_k)  # rho_k es un escalar

            # Actualización de Hk usando la fórmula BFGS
            

            Hk = (I - rho_k.item() * np.outer(s_k, y_k)) @ H0 @ (I - rho_k.item() * np.outer(y_k, s_k)) + rho_k.item() * np.outer(s_k, s_k)

            # Actualizar H0 y x0 para la siguiente iteración
            H0 = Hk
            x0 = xk

            # Calcular la dirección de descenso
            p_k = -H0 @ self.func.grad(x0)

            # Calcular el tamaño de paso alpha usando backtracking line search
            alpha = self.stepCond(x0, p_k, self.func, self.wolfe, condition)
            # Actualizar xk
            xk = x0 + alpha * p_k

            # Guardar el historial de x
            self.x_history.append(x0)
            iterations += 1

        # Convertir el historial a un array de NumPy
        self.x_history = np.array(self.x_history)
        # print(iter)
        return x0

    def plot2D(self):
        x1 = np.linspace(-30, 30, 100)
        x2 = np.linspace(-30, 30, 100)
        X1, X2 = np.meshgrid(x1, x2)
        f = np.zeros((100, 100))
        self.func.d = 2
        self.func.e = np.ones(2).T
        for i in range(100):
            for j in range(100):
                point = np.matrix([[X1[i, j]], [X2[i, j]]])
                f[i, j] = self.func.eval(point)
                # f[i, j] = self.func.eval([X1[i, j], X2[i, j]])

        fig, ax = plt.subplots()
        ax.contour(X1, X2, f, levels=np.linspace(np.min(f), np.max(f), 25))
        plt.xlabel('x1')
        plt.ylabel('x2')
        ax.set_aspect('equal')

        # Dibujar la trayectoria en rojo
        plt.plot(self.x_history[:, 0], self.x_history[:, 1], 'r.', label='Iteraciones')

        # Marcar el último punto con un triángulo azul
        plt.scatter(self.x_history[-1, 0], self.x_history[-1, 1], color='blue', marker='^', s=100, label='Último Punto')

        plt.legend()
        plt.show()
        self.x_history = []

class WolfeConditions:
    def __init__(self, c1=1e-4, c2=0.9):
        self.c1 = c1
        self.c2 = c2
        self.ctd = 0

    def armijo_condition(self, func, x, alpha, d):
        lhs = func.eval(x + alpha * d)
        rhs = func.eval(x)+ self.c1 * alpha * np.dot(func.grad(x).T, d)
        return lhs <= rhs

    def curvature_condition(self, func, x, alpha, d):
        lhs = np.dot(func.grad(x + alpha * d).T, d)
        rhs = self.c2 * np.dot(func.grad(x).T, d)

        return lhs.item() >= rhs.item()

    def strong_wolfe_condition(self, func, x, alpha, d):
        grad_x = func.grad(x)
        grad_x_alpha = func.grad(x + alpha * d)
        lhs = abs(np.dot(grad_x_alpha.T, d))
        rhs = self.c2 * abs(np.dot(grad_x.T, d))

        return lhs.item() <= rhs.item()

# Definición de funciones externas
def stop_criterion(grad_norm, iterations):
    grad_tolerance = 1e-3
    max_iter = 500

    grad_small = grad_norm < grad_tolerance
    iter_exceeded = iterations > max_iter

    return grad_small or iter_exceeded


def step_condition(x, d, func, wolfe, condition):
    alpha = 1.0
    iterations = 0
    min_alpha = 1e-10  # Valor mínimo para alpha
    max_iterations = 50  # Máximo número de iteraciones
    
    if condition == "armijo":
        while not wolfe.armijo_condition(func, x, alpha, d):
            alpha *= 0.5
            iterations += 1
            # print(f"Backtracking: alpha = {alpha}, iteración {iterations}")
            if alpha < min_alpha or iterations >= max_iterations:
                # print(f"WARNING: Backtracking terminado con alpha = {alpha}")
                break

    elif condition == "curvature":
        while not (wolfe.armijo_condition(func, x, alpha, d) and
                   wolfe.curvature_condition(func, x, alpha, d)):
            alpha *= 0.5
            iterations += 1
            if alpha < min_alpha or iterations >= max_iterations:
                # print(f"WARNING: Backtracking terminado con alpha = {alpha}")
                break
                
    elif condition == "strong":
        # print("Iniciando strong condition check...")
        while not (wolfe.armijo_condition(func, x, alpha, d) and
                   wolfe.strong_wolfe_condition(func, x, alpha, d)):
            
            alpha *= 0.5
            iterations += 1
            if alpha < min_alpha or iterations >= max_iterations:
                # print(f"WARNING: Backtracking terminado con alpha = {alpha}")
                break
    else:
        raise ValueError("Condición no soportada. Use 'armijo', 'curvature' o 'strong'.")
    
    return alpha


