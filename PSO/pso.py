from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import animation
import os

def ackley_fun(x):
    """Ackley function
    Domain: -32 < xi < 32
    Global minimum: f_min(0,..,0)=0
    """
    return -20 * np.exp(-.2*np.sqrt(.5*(x[0]**2 + x[1]**2))) - np.exp(.5*(np.cos(np.pi*2*x[0])+np.cos(np.pi*2*x[1]))) + np.exp(1) + 20

def rosenbrock_fun(x):
    """Rosenbrock function
    Domain: -5 < xi < 5
    Global minimun: f_min(1,..,1)=0
    """
    return 100*(x[1] - x[0]**2)**2 + (x[0]-1)**2


def adaptive_pso(func, bounds, swarm_size=10, 
                inertia_start=0.9, inertia_end=0.2, 
                pa_start=0.5, pa_end=2.5,
                ga_start=2.5, ga_end=0.5,
                max_vnorm_start=None, max_vnorm_end=None,
                stagnation_threshold=10, diversity_threshold=0.01,
                num_iters=100, early_stopping=True, 
                verbose=False, func_name=None):
    """Adaptive Particle Swarm Optimization (PSO)
    # Arguments
        func: function to be optimized
        bounds: list, bounds of each dimension
        swarm_size: int, the population size of the swarm
        inertia_start/end: float, initial and final values for inertia weight
        pa_start/end: float, initial and final values for personal acceleration
        ga_start/end: float, initial and final values for global acceleration
        max_vnorm_start/end: float, initial and final values for max velocity norm
        stagnation_threshold: int, number of iterations without improvement to trigger adaptation
        diversity_threshold: float, minimum diversity threshold to trigger parameter changes
        num_iters: int, the maximum number of iterations
        early_stopping: boolean, whether to stop early if convergence is detected
        verbose: boolean, whether to print results or not
        func_name: the name of object function to optimize

    # Returns
        history: history of particles, global bests, and parameter adaptations
    """
    bounds = np.array(bounds)
    assert np.all(bounds[:,0] < bounds[:,1]) # each boundaries have to satisfy this condition
    dim = len(bounds)
    
    # Set default max velocity norms if not provided
    if max_vnorm_start is None:
        max_vnorm_start = 0.2 * np.mean(bounds[:,1] - bounds[:,0])
    if max_vnorm_end is None:
        max_vnorm_end = 0.05 * np.mean(bounds[:,1] - bounds[:,0])
    
    # Initialize current parameter values
    inertia_current = inertia_start
    pa_current = pa_start
    ga_current = ga_start
    max_vnorm_current = max_vnorm_start
    
    X = np.random.rand(swarm_size, dim) # range:0~1, domain:(swarm_size,dim)
    print('## Optimize:',func_name)

    def clip_by_norm(x, max_norm):
        norm = np.linalg.norm(x)
        return x if norm <= max_norm else x * max_norm / norm
    
    def calculate_diversity(particles):
        """Calculate swarm diversity as average distance from centroid"""
        centroid = np.mean(particles, axis=0)
        distances = np.sqrt(np.sum((particles - centroid)**2, axis=1))
        return np.mean(distances)

    # --- step 1 : Initialize all particles randomly in the search-space
    particles = X * (bounds[:,1]-bounds[:,0]) + bounds[:,0]
    velocities = np.random.uniform(-max_vnorm_current, max_vnorm_current, (swarm_size, dim))
    personal_bests = np.copy(particles)
    personal_best_fitness = np.array([func(p) for p in particles])
    
    global_best_idx = np.argmin(personal_best_fitness)
    global_best = personal_bests[global_best_idx].copy()
    global_best_fitness = personal_best_fitness[global_best_idx]
    
    history = {
        'particles': [],
        'global_best_fitness': [], 
        'global_best': [[np.inf, np.inf] for _ in range(num_iters)],
        'obj_func': func_name,
        'inertia': [],
        'pa': [],
        'ga': [],
        'max_vnorm': [],
        'diversity': [],
    }
    
    # Variables for adaptive mechanisms
    stagnation_counter = 0
    best_fitness_history = [global_best_fitness]
    improvement_rate = 0
    
    # --- step 2 : Iteration starts
    for i in range(num_iters):
        # Record current state
        history['particles'].append(particles.copy())
        history['global_best_fitness'].append(global_best_fitness)
        history['global_best'][i][0] = global_best[0]
        history['global_best'][i][1] = global_best[1]
        history['inertia'].append(inertia_current)
        history['pa'].append(pa_current)
        history['ga'].append(ga_current)
        history['max_vnorm'].append(max_vnorm_current)
        
        # Calculate diversity for adaptation decisions
        diversity = calculate_diversity(particles)
        history['diversity'].append(diversity)
        
        if verbose and i % 10 == 0: 
            print('iter# {}: Fitness:{:.5f}, Position:{}, Diversity:{:.5f}'.format(
                i, global_best_fitness, global_best, diversity))
        
        # --- step 3 : Evaluate current swarm
        current_fitness = np.array([func(p) for p in particles])
        
        # Update personal bests
        improved_indices = current_fitness < personal_best_fitness
        personal_bests[improved_indices] = particles[improved_indices].copy()
        personal_best_fitness[improved_indices] = current_fitness[improved_indices]
        
        # Update global best
        min_idx = np.argmin(personal_best_fitness)
        if personal_best_fitness[min_idx] < global_best_fitness:
            global_best_idx = min_idx
            global_best = personal_bests[global_best_idx].copy()
            global_best_fitness = personal_best_fitness[global_best_idx]
            stagnation_counter = 0  # Reset stagnation counter on improvement
        else:
            stagnation_counter += 1
        
        # Store fitness for improvement rate calculation
        best_fitness_history.append(global_best_fitness)
        
        # --- Adaptive Parameter Update Strategies ---
        
        # 1. Time-based adaptation (basic schedule)
        progress = i / num_iters
        inertia_time = inertia_start - progress * (inertia_start - inertia_end)
        pa_time = pa_start + progress * (pa_end - pa_start)
        ga_time = ga_start + progress * (ga_end - ga_start)
        max_vnorm_time = max_vnorm_start - progress * (max_vnorm_start - max_vnorm_end)
        
        # 2. Performance-based adaptation
        if len(best_fitness_history) > 5:
            # Calculate improvement rate over last 5 iterations
            improvement_rate = (best_fitness_history[-6] - best_fitness_history[-1]) / max(1e-10, best_fitness_history[-6])
            
            # If stuck in local optimum (stagnation detected)
            if stagnation_counter >= stagnation_threshold:
                # Increase exploration by boosting inertia and velocity
                inertia_perf = min(0.9, inertia_current * 1.2)
                max_vnorm_perf = max_vnorm_current * 1.5
                pa_perf = pa_current * 0.8  # Reduce personal attraction
                ga_perf = ga_current * 0.8  # Reduce global attraction
                stagnation_counter = 0  # Reset counter after adaptation
            else:
                # Normal adaptation based on improvement rate
                inertia_perf = inertia_current * (1.0 - 0.2 * improvement_rate)
                max_vnorm_perf = max_vnorm_current * (1.0 - 0.1 * improvement_rate)
                pa_perf = pa_current * (1.0 + 0.1 * improvement_rate)
                ga_perf = ga_current * (1.0 + 0.1 * improvement_rate)
        else:
            # Default to time-based before we have enough history
            inertia_perf = inertia_time
            pa_perf = pa_time
            ga_perf = ga_time
            max_vnorm_perf = max_vnorm_time
        
        # 3. Diversity-based adaptation
        if diversity < diversity_threshold:
            # Low diversity - increase exploration
            inertia_div = min(0.9, inertia_current * 1.1)
            pa_div = pa_current * 0.9
            ga_div = ga_current * 0.9
            max_vnorm_div = max_vnorm_current * 1.2
        else:
            # Sufficient diversity - focus on exploitation
            inertia_div = max(0.1, inertia_current * 0.95)
            pa_div = pa_current * 1.05
            ga_div = ga_current * 1.05
            max_vnorm_div = max_vnorm_current * 0.95
        
        # 4. Weighted combination of adaptation strategies
        # Weights shift from time-based to performance/diversity-based as iterations progress
        w_time = 1.0 - progress
        w_perf = progress * 0.6
        w_div = progress * 0.4
        
        # Combine and normalize weights
        w_sum = w_time + w_perf + w_div
        w_time /= w_sum
        w_perf /= w_sum
        w_div /= w_sum
        
        # Combine strategies with weights
        inertia_current = w_time * inertia_time + w_perf * inertia_perf + w_div * inertia_div
        pa_current = w_time * pa_time + w_perf * pa_perf + w_div * pa_div
        ga_current = w_time * ga_time + w_perf * ga_perf + w_div * ga_div
        max_vnorm_current = w_time * max_vnorm_time + w_perf * max_vnorm_perf + w_div * max_vnorm_div
        
        # Ensure parameters stay within reasonable bounds
        inertia_current = np.clip(inertia_current, 0.1, 0.9)
        pa_current = np.clip(pa_current, 0.5, 2.5)
        ga_current = np.clip(ga_current, 0.5, 2.5)
        max_vnorm_current = np.clip(max_vnorm_current, 
                                    0.01 * np.mean(bounds[:,1] - bounds[:,0]),
                                    0.5 * np.mean(bounds[:,1] - bounds[:,0]))
        
        # Early stopping check
        if early_stopping and i > 20:
            recent_improvement = (best_fitness_history[-21] - best_fitness_history[-1]) / max(1e-10, best_fitness_history[-21])
            if recent_improvement < 1e-6 and diversity < diversity_threshold / 5:
                if verbose:
                    print(f"Early stopping at iteration {i} due to convergence")
                # Fill remaining history slots with final values
                for j in range(i+1, num_iters):
                    history['particles'].append(particles.copy())
                    history['global_best_fitness'].append(global_best_fitness)
                    history['global_best'][j][0] = global_best[0]
                    history['global_best'][j][1] = global_best[1]
                    history['inertia'].append(inertia_current)
                    history['pa'].append(pa_current)
                    history['ga'].append(ga_current)
                    history['max_vnorm'].append(max_vnorm_current)
                    history['diversity'].append(diversity)
                break
        
        # --- step 4 : Calculate the acceleration and momentum with updated parameters
        m = inertia_current * velocities
        
        # Random coefficients for each particle independently
        r1 = np.random.rand(swarm_size, dim)
        r2 = np.random.rand(swarm_size, dim)
        
        acc_local = pa_current * r1 * (personal_bests - particles)
        acc_global = ga_current * r2 * (global_best - particles)
        
        # --- step 5 : Update the velocities
        velocities = m + acc_local + acc_global
        
        # Apply velocity bounds for each particle
        for idx in range(swarm_size):
            velocities[idx] = clip_by_norm(velocities[idx], max_vnorm_current)
        
        # --- step 6 : Update the position of particles
        particles = particles + velocities
        
        # --- step 7: Ensure particles stay within bounds
        particles = np.clip(particles, bounds[:, 0], bounds[:, 1])

    return history


def visualizeHistory2D(func=None, history=None, bounds=None, 
                      minima=None, func_name='', save2mp4=False, save2gif=False):
    """Visualize the process of optimizing
    # Arguments
        func: object function
        history: dict, object returned from pso above
        bounds: list, bounds of each dimention
        minima: list, the exact minima to show in the plot
        func_name: str, the name of the object function
        save2mp4: bool, whether to save as mp4 or not
    """

    print('## Visualizing optimizing {}'.format(func_name))
    assert len(bounds)==2

    # define meshgrid according to given boundaries
    x = np.linspace(bounds[0][0], bounds[0][1], 50)
    y = np.linspace(bounds[1][0], bounds[1][1], 50)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func([X[i, j], Y[i, j]])

    # initialize figure
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(231, facecolor='w')
    ax2 = fig.add_subplot(232, facecolor='w')
    ax3 = fig.add_subplot(233, facecolor='w')
    ax4 = fig.add_subplot(234, facecolor='w')
    ax5 = fig.add_subplot(235, facecolor='w')
    ax6 = fig.add_subplot(236, facecolor='w')

    # animation callback function
    def animate(frame, history):
        # Clear axes
        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax4.cla()
        ax5.cla()
        ax6.cla()
        
        # Set up main particle visualization
        ax1.set_xlabel('X1')
        ax1.set_ylabel('X2')
        ax1.set_title('{}|iter={}|Gbest=({:.5f},{:.5f})'.format(func_name, frame+1,
                      history['global_best'][frame][0], history['global_best'][frame][1]))
        ax1.set_xlim(bounds[0][0], bounds[0][1])
        ax1.set_ylim(bounds[1][0], bounds[1][1])
        
        # Set up fitness plot
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Fitness (log scale)')
        ax2.set_title('Minima Value Plot|Population={}|MinVal={:.6e}'.format(
            len(history['particles'][0]), history['global_best_fitness'][frame]))
        ax2.set_xlim(0, len(history['global_best_fitness']))
        ax2.set_ylim(10e-16, 10e1)
        ax2.set_yscale('log')
        
        # Set up inertia plot
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Inertia Value')
        ax3.set_title('Inertia Adaptation')
        ax3.set_xlim(0, len(history['inertia']))
        ax3.set_ylim(0, 1)
        
        # Set up acceleration plots
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Parameter Value')
        ax4.set_title('Acceleration Parameters')
        ax4.set_xlim(0, len(history['pa']))
        ax4.set_ylim(0, 3)
        
        # Set up max velocity plot
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Max Velocity')
        ax5.set_title('Max Velocity Adaptation')
        ax5.set_xlim(0, len(history['max_vnorm']))
        
        # Set up diversity plot
        ax6.set_xlabel('Iteration')
        ax6.set_ylabel('Diversity')
        ax6.set_title('Swarm Diversity')
        ax6.set_xlim(0, len(history['diversity']))
        ax6.set_yscale('log')

        # data to be plot
        data = history['particles'][frame]
        global_best = np.array(history['global_best_fitness'])

        # contour and global minimum
        contour = ax1.contour(X, Y, Z, levels=50, cmap="magma")
        ax1.plot(minima[0], minima[1], marker='o', color='black')

        # plot particles
        ax1.scatter(data[:,0], data[:,1], marker='x', color='black')
        if frame > 1:
            for i in range(len(data)):
                ax1.plot([history['particles'][frame-n][i][0] for n in range(2,-1,-1)],
                         [history['particles'][frame-n][i][1] for n in range(2,-1,-1)])
        elif frame == 1:
            for i in range(len(data)):
                ax1.plot([history['particles'][frame-n][i][0] for n in range(1,-1,-1)],
                         [history['particles'][frame-n][i][1] for n in range(1,-1,-1)])

        # plot current global best
        x_range = np.arange(0, frame+1)
        ax2.plot(x_range, global_best[0:frame+1])
        
        # plot parameter adaptations
        if 'inertia' in history and len(history['inertia']) > 0:
            ax3.plot(x_range, history['inertia'][0:frame+1])
            
        if 'pa' in history and 'ga' in history and len(history['pa']) > 0:
            ax4.plot(x_range, history['pa'][0:frame+1], label='Personal')
            ax4.plot(x_range, history['ga'][0:frame+1], label='Global')
            ax4.legend()
            
        if 'max_vnorm' in history and len(history['max_vnorm']) > 0:
            ax5.plot(x_range, history['max_vnorm'][0:frame+1])
            
        if 'diversity' in history and len(history['diversity']) > 0:
            ax6.plot(x_range, history['diversity'][0:frame+1])
        
    # title of figure
    fig.suptitle('Adaptive PSO Optimization of {} function, f_min({},{})={}'.format(
        func_name.split()[0], minima[0], minima[1], func(minima)), fontsize=20)

    ani = animation.FuncAnimation(fig, animate, fargs=(history,),
                    frames=len(history['particles']), interval=250, repeat=False, blit=False)

    ## Save animation if requested
    if save2mp4:
        os.makedirs('mp4/', exist_ok=True)
        ani.save('mp4/Adaptive_PSO_{}_population_{}.mp4'.format(
            func_name.split()[0], len(history['particles'][0])), writer="ffmpeg", dpi=100)
        print('A mp4 video is saved at mp4/')
    elif save2gif:
        os.makedirs('gif/', exist_ok=True)
        ani.save('gif/Adaptive_PSO_{}_population_{}.gif'.format(
            func_name.split()[0], len(history['particles'][0])), writer="imagemagick")
        print('A gif video is saved at gif/')
    else:
        plt.show()


# Example usage
if __name__ == "__main__":
    # Ackley function
    history_ackley = adaptive_pso(
        func=ackley_fun, 
        bounds=[[-32,32],[-32,32]], 
        swarm_size=30, 
        inertia_start=0.9, inertia_end=0.2,
        pa_start=0.5, pa_end=2.5,
        ga_start=2.5, ga_end=0.5,
        stagnation_threshold=10,
        num_iters=50, 
        verbose=True, 
        func_name='Ackley Function'
    )
    print('Ackley global best fitness:', history_ackley['global_best_fitness'][-1])
    print('Ackley global best position:', history_ackley['global_best'][-1])
    visualizeHistory2D(
        func=ackley_fun, 
        history=history_ackley, 
        bounds=[[-32,32],[-32,32]], 
        minima=[0,0], 
        func_name='Ackley Function'
    )

    # Rosenbrock function
    history_rosenbrock = adaptive_pso(
        func=rosenbrock_fun, 
        bounds=[[-2,2],[-2,2]], 
        swarm_size=30, 
        inertia_start=0.9, inertia_end=0.2,
        pa_start=0.5, pa_end=2.5,
        ga_start=2.5, ga_end=0.5,
        num_iters=100, 
        verbose=True, 
        func_name='Rosenbrock Function'
    )
    print('Rosenbrock global best fitness:', history_rosenbrock['global_best_fitness'][-1])
    print('Rosenbrock global best position:', history_rosenbrock['global_best'][-1])
    visualizeHistory2D(
        func=rosenbrock_fun, 
        history=history_rosenbrock, 
        bounds=[[-2,2],[-2,2]], 
        minima=[1,1], 
        func_name='Rosenbrock Function'
    )