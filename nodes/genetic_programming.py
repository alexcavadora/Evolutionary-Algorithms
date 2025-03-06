from __future__ import annotations

import random
import statistics
import copy
import os
import sys
from typing import Callable, List, Union, Optional

import matplotlib.pyplot as plt

try:
    import graphviz
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False
    print("Graphviz not installed. Tree visualization will be text-based.", file=sys.stderr)

from node import BlifDecoder  # Add BlifDecoder import

# Type aliases
Terminal = Union[str, int, float]
Function = Callable[[float, float], float]

# Configuration constants
POP_SIZE: int = 20        # population size
MIN_DEPTH: int = 2        # minimal initial random tree depth
MAX_DEPTH: int = 5        # maximal initial random tree depth
GENERATIONS: int = 250    # maximal number of generations to run evolution
TOURNAMENT_SIZE: int = 5  # size of tournament for tournament selection
XO_RATE: float = 0.8      # crossover rate 
PROB_MUTATION: float = 0.2  # per-node mutation probability 
BLOAT_CONTROL: bool = False  # True adds bloat control to fitness function

BLIF_FILE: str = "dec.blif"  # Path to your BLIF file
OPTIMIZATION_MODE: str = "circuit"  # "circuit" or "math"

def invert_bit(x: float, y: float) -> float:
    """Invert a bit value (NOT gate). Ignores y."""
    return 1.0 - x

def and_gate(x: float, y: float) -> float:
    """AND gate function."""
    return float(x > 0.5 and y > 0.5)

def or_gate(x: float, y: float) -> float:
    """OR gate function."""
    return float(x > 0.5 or y > 0.5)

def xor_gate(x: float, y: float) -> float:
    """XOR gate function."""
    return float((x > 0.5) != (y > 0.5))

def nand_gate(x: float, y: float) -> float:
    """NAND gate function."""
    return invert_bit(and_gate(x, y), 0.0)

def nor_gate(x: float, y: float) -> float:
    """NOR gate function."""
    return invert_bit(or_gate(x, y), 0.0)

# Replace original function and terminal definitions
# Define math functions FIRST
def add(x: float, y: float) -> float: 
    return x + y

def sub(x: float, y: float) -> float: 
    return x - y

def mul(x: float, y: float) -> float: 
    return x * y

# Then declare the function lists
CIRCUIT_FUNCTIONS: List[Function] = [and_gate, or_gate, nor_gate, nand_gate, invert_bit]
MATH_FUNCTIONS: List[Function] = [add, sub, mul]

# Default terminals based on mode
FUNCTIONS = CIRCUIT_FUNCTIONS if OPTIMIZATION_MODE == "circuit" else MATH_FUNCTIONS
TERMINALS: List[Terminal] = ['x', 0, 1] if OPTIMIZATION_MODE == "circuit" else ['x', -2, -1, 0, 1, 2]

# Initialize BLIF decoder
blif_decoder = BlifDecoder(BLIF_FILE) if OPTIMIZATION_MODE == "circuit" else None

def generate_test_vectors(num_vectors: int = 20) -> List[List[int]]:
    """Generate random test vectors for circuit evaluation."""
    input_size = len(blif_decoder.inputs)
    return [
        [random.randint(0, 1) for _ in range(input_size)]
        for _ in range(num_vectors)
    ]

TEST_VECTORS = generate_test_vectors() if OPTIMIZATION_MODE == "circuit" else []

def add(x: float, y: float) -> float: 
    return x + y

def sub(x: float, y: float) -> float: 
    return x - y

def mul(x: float, y: float) -> float: 
    return x * y

FUNCTIONS: List[Function] = [add, sub, mul]
TERMINALS: List[Terminal] = ['x', -2, -1, 0, 1, 2]

def target_func(x: float) -> float:
    """Evolution's target function."""
    return x**4 + x**3 + x**2 + x + 1

def generate_dataset() -> List[List[float]]:
    """Generate dataset from target function."""
    return [[x/100, target_func(x/100)] for x in range(-100, 101, 2)]

class GPTree:
    def __init__(
        self, 
        data: Optional[Union[Function, Terminal]] = None, 
        left: Optional["GPTree"] = None, 
        right: Optional["GPTree"] = None
    ):
        self.data: Optional[Union[Function, Terminal]] = data
        self.left: Optional[GPTree] = left
        self.right: Optional[GPTree] = right
        
    def node_label(self) -> str:
        """Return string label for the node."""
        if callable(self.data):
            return self.data.__name__
        return str(self.data)
    
    def draw(self, dot: graphviz.Digraph, count: List[int]) -> None:
        """Draw the tree using Graphviz."""
        node_name = str(count[0])
        dot.node(node_name, self.node_label())
        
        if self.left:
            count[0] += 1
            dot.edge(node_name, str(count[0]))
            self.left.draw(dot, count)
        
        if self.right:
            count[0] += 1
            dot.edge(node_name, str(count[0]))
            self.right.draw(dot, count)
        
    def text_tree_repr(self, indent: str = '') -> str:
        """Create a text representation of the tree."""
        if not callable(self.data):
            return f"{indent}{self.node_label()}"
        
        repr_str = f"{indent}{self.node_label()}\n"
        if self.left:
            repr_str += self.left.text_tree_repr(indent + '  ') + '\n'
        if self.right:
            repr_str += self.right.text_tree_repr(indent + '  ')
        return repr_str

    def draw_tree(self, fname: str, footer: str) -> None:
        """Render and display the tree."""
        # Text-based fallback visualization
        print(f"\n{footer}")
        print(self.text_tree_repr())

        # Graphviz visualization if available
        if HAS_GRAPHVIZ:
            try:
                dot = graphviz.Digraph()
                dot.graph_attr['label'] = footer
                count = [0]
                self.draw(dot, count)
                
                # Save and display the tree
                dot.render(f"{fname}.gv", format="png", view=False)
                plt.figure()
                plt.imshow(plt.imread(f"{fname}.gv.png"))
                plt.axis('off')
                plt.show()
            except Exception as e:
                print(f"Graphviz visualization failed: {e}", file=sys.stderr)
    def compute_tree(self, x: float) -> float:
        """Compute the value of the tree for a given x."""
        if callable(self.data):
            return self.data(
                self.left.compute_tree(x) if self.left else 0, 
                self.right.compute_tree(x) if self.right else 0
            )
        elif self.data == 'x':
            return x
        elif isinstance(self.data, (int, float)):
            return self.data
        raise ValueError(f"Invalid node data: {self.data}")
            
    def random_tree(
        self, 
        grow: bool, 
        max_depth: int, 
        depth: int = 0
    ) -> None:
        """Create a random tree using grow or full method."""
        # Determine node type based on depth and randomness
        if depth < MIN_DEPTH or (depth < max_depth and not grow):
            self.data = random.choice(FUNCTIONS)
        elif depth >= max_depth:   
            self.data = random.choice(TERMINALS)
        else:
            self.data = (random.choice(TERMINALS) if random.random() > 0.5 
                         else random.choice(FUNCTIONS))
        
        # Recursively create children if the node is a function
        if callable(self.data):
            self.left = GPTree()          
            self.left.random_tree(grow, max_depth, depth=depth+1)            
            self.right = GPTree()
            self.right.random_tree(grow, max_depth, depth=depth+1)

    def mutation(self) -> None:
        """Mutate the tree with a given probability."""
        if random.random() < PROB_MUTATION:
            # Si el nodo original es una función, aseguramos que el mutado también sea función
            if self.data in FUNCTIONS:
                self.data = random.choice(FUNCTIONS)
                self.left = GPTree()
                self.right = GPTree()
                self.left.random_tree(grow=True, max_depth=2)
                self.right.random_tree(grow=True, max_depth=2)
            else:  
                # Si es un terminal, aseguramos que siga siendo terminal
                self.data = random.choice(TERMINALS)
        else:
            # Recursivamente mutar hijos
            if self.left:
                self.left.mutation()
            if self.right:
                self.right.mutation()

        
    def size(self) -> int:
        """Calculate the size of the tree in nodes."""
        if not callable(self.data):
            return 1
        l_size = self.left.size() if self.left else 0
        r_size = self.right.size() if self.right else 0
        return 1 + l_size + r_size

    def build_subtree(self) -> GPTree:
        """Create a deep copy of the subtree."""
        t = GPTree()
        t.data = self.data
        t.left = self.left.build_subtree() if self.left else None
        t.right = self.right.build_subtree() if self.right else None
        return t
                        
    def scan_tree(
        self, 
        count: List[int], 
        second: Optional[GPTree] = None
    ) -> Optional[GPTree]:
        """Scan and potentially modify the tree."""
        count[0] -= 1            
        if count[0] <= 1: 
            if second is None:
                # Return subtree rooted here
                return self.build_subtree()
            else:
                # Glue subtree here
                self.data = second.data
                self.left = second.left
                self.right = second.right
                return None
        
        ret = None              
        if self.left and count[0] > 1:
            ret = self.left.scan_tree(count, second)  
        if self.right and count[0] > 1:
            ret = self.right.scan_tree(count, second)  
        return ret

    def crossover(self, other: GPTree) -> None:
        """Perform crossover between two trees."""
        if random.random() < XO_RATE:
            # Select a random subtree from the other tree
            second = other.scan_tree([random.randint(1, other.size())], None)
            
            # Replace a random subtree in this tree with the selected subtree
            if second:
                self.scan_tree([random.randint(1, self.size())], second)
                if second is not None:
                    self.scan_tree([random.randint(1, self.size())], second)


def init_population() -> List[GPTree]:
    """Mejora la inicialización de la población."""
    pop = []
    for md in range(3, MAX_DEPTH + 1):
        for _ in range(POP_SIZE // 4):  # Reducir tamaño de cada grupo
            t = GPTree()
            t.random_tree(grow=True, max_depth=md)
            pop.append(t)

            t = GPTree()
            t.random_tree(grow=False, max_depth=md)
            pop.append(t)
    return pop


def error(individual: GPTree, dataset: List[List[float]]) -> float:
    """Calculate mean absolute error (MAE) with error handling."""
    if not dataset:
        return float('inf')  # Retorna un error infinito si no hay datos

    errors = []
    for ds in dataset:
        if len(ds) != 2:  # Verificar que los datos sean correctos
            print(f"Error: dataset mal formado {ds}", file=sys.stderr)
            continue  # Ignorar este punto

        x, y = ds
        try:
            prediction = individual.compute_tree(x)
            if not isinstance(prediction, (int, float)):  # Verifica que compute_tree(x) sea válido
                raise ValueError(f"compute_tree devolvió un valor no numérico: {prediction}")
            if abs(prediction) > 1e6:  # Evitar valores extremos
                print(f"Advertencia: Predicción muy grande {prediction} para x={x}", file=sys.stderr)
                prediction = float('inf')

            errors.append(abs(prediction - y))
        except Exception as e:
            print(f"Error en la evaluación de {ds}: {e}", file=sys.stderr)
            errors.append(float('inf'))  # Penaliza errores con un valor alto
    
    return statistics.mean(errors) if errors else float('inf')  # Maneja el caso de lista vacía

def circuit_error(individual: GPTree) -> float:
    """Calculate error for circuit optimization."""
    total_error = 0
    for test_vector in TEST_VECTORS:
        # Set inputs and simulate original circuit
        blif_decoder.set_inputs_from_vector(test_vector)
        blif_decoder.simulate()
        original_outputs = blif_decoder.get_outputs_as_vector()
        
        # Apply GP tree modifications
        modified_outputs = [
            1 if individual.compute_tree(output) > 0.5 else 0 
            for output in original_outputs
        ]
        
        # Calculate target outputs (example: invert all outputs)
        target_outputs = [1 - output for output in original_outputs]
        
        # Accumulate error
        total_error += sum(m != t for m, t in zip(modified_outputs, target_outputs))
    
    return total_error / (len(TEST_VECTORS) * len(target_outputs))

def fitness(individual: GPTree, dataset: Optional[List[List[float]]] = None) -> float:
    """Updated fitness function for circuit optimization."""
    if OPTIMIZATION_MODE == "circuit":
        err = circuit_error(individual)
    else:
        err = error(individual, dataset or generate_dataset())
    
    if BLOAT_CONTROL:
        return 1 / (1 + err + 0.05 * individual.size())
    return 1 / (1 + err)
                
def selection(population: List[GPTree], fitnesses: List[float]) -> GPTree:
    """Mejor selección por torneo."""
    k = min(TOURNAMENT_SIZE, len(population))
    tournament_indices = random.choices(range(len(population)), k=k)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]

    # Priorizar individuos con mejor fitness
    best_index = max(range(len(tournament_indices)), key=lambda i: tournament_fitnesses[i])
    return copy.deepcopy(population[tournament_indices[best_index]])


def prepare_plots() -> tuple:
    """Prepare interactive matplotlib plots."""
    plt.ion()  # Interactive mode
    fig, (ax_error, ax_size) = plt.subplots(2, sharex=True)
    fig.suptitle('EVOLUTIONARY PROGRESS')
    
    ax_error.set_title('Error', fontsize=14)
    ax_size.set_title('Mean Tree Size', fontsize=14)
    plt.xlabel('Generation', fontsize=12)
    
    ax_error.set_xlim(0, GENERATIONS)
    ax_error.set_ylim(0, 1)
    
    line_error, = ax_error.plot([], [], 'b-')
    line_size, = ax_size.plot([], [], 'r-')
    
    return fig, ax_error, ax_size, line_error, line_size

def main():
    # Initialize random seed
    random.seed()
    
    # Prepare dataset and initial population
    dataset = generate_dataset()
    population = init_population()
    
    # Prepare tracking variables
    best_of_run = None
    best_of_run_error = float('inf')
    best_of_run_gen = 0
    
    # Compute initial fitnesses
    fitnesses = [fitness(ind, dataset) for ind in population]
    
    # Prepare plotting
    fig, ax_error, ax_size, line_error, line_size = prepare_plots()
    
    # Track generations for plotting
    gen_data = []
    error_data = []
    size_data = []
    max_mean_size = 0
    
    # Evolution loop
    for gen in range(GENERATIONS):
        # Create next generation
        next_gen_population = []
        for _ in range(POP_SIZE):
            parent1 = selection(population, fitnesses)
            parent2 = selection(population, fitnesses)
            parent1.crossover(parent2)
            parent1.mutation()
            next_gen_population.append(parent1)
        
        population = next_gen_population
        
        # Compute fitness and errors
        fitnesses = [fitness(ind, dataset) for ind in population]
        errors = [error(ind, dataset) for ind in population]
        
        # Track best individual
        current_best_error = min(errors)
        if current_best_error < best_of_run_error:
            best_of_run_error = current_best_error
            best_of_run_gen = gen
            best_of_run = copy.deepcopy(population[errors.index(current_best_error)])
            
            print(f"________________________")
            best_of_run.draw_tree(
                "best_of_run", 
                f"gen: {gen}, error: {best_of_run_error:.3f}"
            )
        
        # Track data for plotting
        gen_data.append(gen)
        error_data.append(current_best_error)
        
        # Compute and track tree sizes
        sizes = [ind.size() for ind in population]
        mean_size = statistics.mean(sizes)
        size_data.append(mean_size)
        
        # Update plots
        ax_error.clear()
        ax_size.clear()
        ax_error.plot(gen_data, error_data, 'b-')
        ax_size.plot(gen_data, size_data, 'r-')
        ax_error.set_title('Error')
        ax_size.set_title('Mean Tree Size')
        plt.pause(0.01)
        
        # Early stopping
        if best_of_run_error <= 1e-5:
            break
    
    # Final output
    print(f"END OF RUN (bloat control was {'ON' if BLOAT_CONTROL else 'OFF'})")
    print(f"\nBest solution found at generation {best_of_run_gen}")
    print(f"Best solution error: {best_of_run_error:.6f}")
    
    # Draw final best tree
    if best_of_run:
        best_of_run.draw_tree(
            "best_of_run", 
            f"Final solution at gen {best_of_run_gen}, error: {best_of_run_error:.6f}"
        )
    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()