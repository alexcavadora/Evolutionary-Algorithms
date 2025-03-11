import random
from typing import Callable, List, Union
from graphviz import Digraph
import uuid

# Type aliases
Terminal = Union[str, int, float]
Function = Callable[[float, float], float]

# Circuit optimization functions
def and_gate(x: float, y: float) -> float:
    """AND gate function."""
    return float(x > 0.5 and y > 0.5)

def or_gate(x: float, y: float) -> float:
    """OR gate function."""
    return float(x > 0.5 or y > 0.5)

def nand_gate(x: float, y: float) -> float:
    """NAND gate function."""
    return float(not (x > 0.5 and y > 0.5))

def nor_gate(x: float, y: float) -> float:
    """NOR gate function."""
    return float(not (x > 0.5 or y > 0.5))


# Then declare the function lists
CIRCUIT_FUNCTIONS: List[Function] = [and_gate, or_gate, nor_gate, nand_gate]
TERMINALS: List[Terminal] = [0,1,2,3,4,5,6,7]

# random.seed(42)

class node:

    def __init__(self):
        self.value = None
        self.left = None
        self.right = None
    
    def copy(self):
        n = node()
        n.value = self.value
        if self.left:
            n.left = self.left.copy()
        if self.right:
            n.right = self.right.copy()
        return n

    def rand_value(self, p_terminals):
        u = random.uniform(0, 1)
        if u < p_terminals:
            return random.choice(CIRCUIT_FUNCTIONS)
        else:
            return random.choice(TERMINALS)
    
    def rand_tree(self, depth , p_terminals=1.0):
        if depth == 0:
            n = node()
            n.value = random.choice(TERMINALS)
            return n
        else:
            n = node() 
            n.value = self.rand_value(p_terminals)
            if type(n.value) != int:
                n.left = self.rand_tree(depth - 1, p_terminals-0.2)
                n.right = self.rand_tree(depth - 1, p_terminals-0.2)
            return n
    
    def get_depth(self):
        if type(self.value) == int:
            return 1
        else:
            return 1 + max(self.left.get_depth(), self.right.get_depth())

    def swap_point(self, p_swap=0.15):
        u = random.uniform(0, 1)
        if u < p_swap or type(self.value) == int:
            return self 
        else:
            if self.left:
                self.left.swap_point(p_swap+0.15)
            if self.right:
                self.right.swap_point(p_swap+0.15)

    def swap_subtrees(self, cp, p_swap=0.2):
        u = random.uniform(0, 1)
        if u < p_swap or type(self.value) == int:
            # self.value = cp.value
            # self.left = cp.left
            # self.right = cp.right
            self = cp
        else:
            if self.left:
                self.left.swap_subtrees(cp, p_swap+0.1)
            if self.right:
                self.right.swap_subtrees(cp, p_swap+0.1)

    def swap(self, tree):
        father1 = self.copy()
        father2 = tree.copy()
        cp1 = father1.swap_point()
        cp2 = father2.swap_point()
        father1.swap_subtrees(cp2)
        father2.swap_subtrees(cp1)

        return father1, father2

    def mutation(self,depht, p_mutation=0.1):
        u = random.uniform(0, 1)
        if u < p_mutation or type(self.value) == int:
            if random.uniform(0, 1) < 0.5:
                self.left = self.rand_tree(depht-1)
            else:
                self.right = self.rand_tree(depht-1)
        else:
            if self.left:
                self.left.mutation(depht-1,p_mutation+0.1)
            if self.right:
                self.right.mutation(depht-1,p_mutation+0.1)

    def evaluate(self, inputs):
        if type(self.value) == int:
            return inputs[self.value]
        else:
            # print(self.value(1,1))
            return self.value(self.left.evaluate(inputs), self.right.evaluate(inputs))
    
    def print_tree(self):
        if type(self.value) == int:
            print(self.value)
        else:
            print(self.value.__name__)
            self.left.print_tree()
            self.right.print_tree()
    
    def visualize(self, filename='trees/my_circuit_tree'):
        """
        Visualiza el árbol usando Graphviz
        Args:
            filename: Nombre del archivo de salida sin extensión
        """
        
        dot = Digraph(comment='Árbol de Circuito')
        
        def add_nodes_edges(node, parent_id=None):
            if node is None:
                return
            
            # Generar ID único para el nodo actual
            node_id = str(uuid.uuid4())
            
            # Determinar la etiqueta y el color del nodo
            if type(node.value) == int:
                label = f"X[{node.value}]"
                color = "lightblue"
            else:
                label = node.value.__name__
                color = "lightgreen"
            
            # Añadir el nodo al grafo
            dot.node(node_id, label, style='filled', fillcolor=color)
            
            # Si tiene padre, añadir la conexión
            if parent_id:
                dot.edge(parent_id, node_id)
            
            # Procesar hijos recursivamente
            if node.left:
                add_nodes_edges(node.left, node_id)
            if node.right:
                add_nodes_edges(node.right, node_id)
        
        # Comenzar el proceso recursivo desde la raíz
        add_nodes_edges(self)
        
        # Renderizar y guardar el grafo
        dot.render(filename, view=True, format='png')
        
        return dot

if __name__ == "__main__":
    # Ejemplo de uso
    root = node()
    root = root.rand_tree(depth=3)
    print("Árbol original:")
    root.print_tree()
    root.visualize(filename='trees/original_tree')

    root.mutation(3)
    print("\nÁrbol después de la mutación:")
    root.print_tree()
    root.visualize(filename='trees/mutated_tree')