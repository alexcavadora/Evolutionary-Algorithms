import networkx as nx
import matplotlib.pyplot as plt
import re
from collections import defaultdict

def parse_blif(file_content):
    """Parse BLIF file content into a structured representation"""
    lines = file_content.strip().split('\n')
    
    # Initialize data structures
    model_name = None
    inputs = []
    outputs = []
    gates = []
    connections = []
    current_gate = None
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        if line.startswith('.model'):
            model_name = line.split()[1]
        elif line.startswith('.inputs'):
            inputs.extend(re.findall(r'\S+', line[7:]))
        elif line.startswith('.outputs'):
            outputs.extend(re.findall(r'\S+', line[8:]))
        elif line.startswith('.names'):
            parts = line.split()
            gate_inputs = parts[1:-1]
            gate_output = parts[-1]
            current_gate = {
                'name': gate_output,
                'inputs': gate_inputs,
                'truth_table': []
            }
            gates.append(current_gate)
        elif line.startswith('.end'):
            break
        elif current_gate is not None and line[0] in '01-':
            current_gate['truth_table'].append(line)
        
        # Capture connections
        for gate in gates:
            for input_name in gate['inputs']:
                connections.append((input_name, gate['name']))
    
    return {
        'model_name': model_name,
        'inputs': inputs,
        'outputs': outputs,
        'gates': gates,
        'connections': connections
    }

def create_circuit_graph(blif_data):
    """Create a NetworkX graph from parsed BLIF data"""
    G = nx.DiGraph()
    
    # Add input nodes
    for input_name in blif_data['inputs']:
        G.add_node(input_name, type='input')
    
    # Add output nodes
    for output_name in blif_data['outputs']:
        G.add_node(output_name, type='output')
    
    # Add gate nodes and connections
    for gate in blif_data['gates']:
        G.add_node(gate['name'], type='gate')
        for input_name in gate['inputs']:
            G.add_edge(input_name, gate['name'])
    
    # Add connections to output nodes
    for conn in blif_data['connections']:
        if conn[1] in blif_data['outputs']:
            G.add_edge(conn[0], conn[1])
    
    return G

def visualize_circuit(blif_content, max_nodes=100, focus_on_inputs=True):
    """Visualize the circuit with a focus on the specified area"""
    # Parse BLIF
    blif_data = parse_blif(blif_content)
    G = create_circuit_graph(blif_data)
    
    # Simplify the graph if it's too large
    if len(G.nodes) > max_nodes:
        if focus_on_inputs:
            # Keep only a subset of nodes connected to inputs
            nodes_to_keep = set(blif_data['inputs'])
            # Add some immediate connections
            for node in list(nodes_to_keep):
                nodes_to_keep.update(list(G.successors(node))[:3])
            
            # Create subgraph
            G = G.subgraph(nodes_to_keep)
        else:
            # Sample a representative portion
            nodes = list(G.nodes)
            sampled_nodes = nodes[:max_nodes]
            G = G.subgraph(sampled_nodes)
    
    # Create node color map
    node_colors = []
    for node in G.nodes:
        if node in blif_data['inputs']:
            node_colors.append('lightblue')
        elif node in blif_data['outputs']:
            node_colors.append('lightgreen')
        else:
            node_colors.append('lightgray')
    
    # Position nodes using a hierarchical layout
    pos = nx.spring_layout(G, seed=42)
    
    # Create the visualization
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, 
            node_size=500, font_size=8, arrows=True,
            connectionstyle='arc3,rad=0.1')
    
    # Add legend
    plt.plot([0], [0], 'o', color='lightblue', label='Inputs')
    plt.plot([0], [0], 'o', color='lightgreen', label='Outputs')
    plt.plot([0], [0], 'o', color='lightgray', label='Gates')
    plt.legend()
    
    plt.title(f"Circuit Visualization: {blif_data['model_name']}")
    plt.tight_layout()
    
    return plt

def analyze_blif(blif_content):
    """Analyze the BLIF content and return interesting statistics"""
    blif_data = parse_blif(blif_content)
    
    # Count gate types
    gate_types = defaultdict(int)
    for gate in blif_data['gates']:
        # Determine gate type based on truth table
        if len(gate['truth_table']) == 1 and gate['truth_table'][0] == '11 1':
            gate_type = 'AND'
        elif len(gate['truth_table']) == 2 and set(gate['truth_table']) == {'01 1', '10 1'}:
            gate_type = 'XOR'
        elif len(gate['truth_table']) == 1 and gate['truth_table'][0] == '1 1':
            gate_type = 'BUFFER'
        elif len(gate['truth_table']) == 1 and gate['truth_table'][0] == '0 1':
            gate_type = 'NOT'
        else:
            gate_type = 'CUSTOM'
        
        gate_types[gate_type] += 1
    
    # Get statistics
    stats = {
        'model_name': blif_data['model_name'],
        'num_inputs': len(blif_data['inputs']),
        'num_outputs': len(blif_data['outputs']),
        'num_gates': len(blif_data['gates']),
        'num_connections': len(blif_data['connections']),
        'gate_types': dict(gate_types)
    }
    
    return stats

def create_structural_visualization(blif_content):
    """Create a structural visualization showing the circuit layers"""
    blif_data = parse_blif(blif_content)
    G = create_circuit_graph(blif_data)
    
    # Assign layers to nodes
    layers = {}
    
    # Inputs are in layer 0
    for node in blif_data['inputs']:
        layers[node] = 0
    
    # Assign layers to other nodes based on longest path from inputs
    remaining_nodes = set(G.nodes) - set(blif_data['inputs'])
    
    # Simple topological sort-based layer assignment
    changed = True
    while changed and remaining_nodes:
        changed = False
        for node in list(remaining_nodes):
            predecessors = list(G.predecessors(node))
            if all(pred in layers for pred in predecessors):
                # Assign layer as max of predecessors + 1
                if predecessors:
                    layers[node] = max(layers[pred] for pred in predecessors) + 1
                else:
                    layers[node] = 1  # No predecessors but not an input
                remaining_nodes.remove(node)
                changed = True
    
    # Handle any remaining nodes (cycles, etc.)
    for node in remaining_nodes:
        layers[node] = max(layers.values()) + 1
    
    # Create a layered visualization
    plt.figure(figsize=(14*3, 10*3))
    
    # Group nodes by layer
    layer_nodes = defaultdict(list)
    for node, layer in layers.items():
        layer_nodes[layer].append(node)
    
    # Determine node positions
    pos = {}
    max_layer = max(layers.values())
    
    for layer, nodes in layer_nodes.items():
        y = 1.0 - (layer / max(1, max_layer))
        for i, node in enumerate(sorted(nodes)):
            x = (i + 1) / (len(nodes) + 1)
            pos[node] = (x, y)
    
    # Draw the graph
    node_colors = []
    for node in G.nodes:
        if node in blif_data['inputs']:
            node_colors.append('lightblue')
        elif node in blif_data['outputs']:
            node_colors.append('lightgreen')
        else:
            node_colors.append('lightsalmon')
    
    nx.draw(G, pos, with_labels=True, node_color=node_colors, 
            node_size=300, font_size=7, arrows=True, 
            edge_color='gray', width=0.5,
            connectionstyle='arc3,rad=0.1')
    
    # Add layer labels
    for layer in layer_nodes:
        plt.text(-0.05, 1.0 - (layer / max(1, max_layer)), f"Layer {layer}", 
                 fontsize=10, ha='right', va='center')
    
    # Add legend
    plt.plot([0], [0], 'o', color='lightblue', label='Inputs')
    plt.plot([0], [0], 'o', color='lightgreen', label='Outputs')
    plt.plot([0], [0], 'o', color='lightsalmon', label='Gates')
    plt.legend(loc='upper right')
    
    plt.title(f"Layered Circuit Structure: {blif_data['model_name']}")
    plt.axis('off')
    plt.tight_layout()
    
    return plt

# Example usage
# Assuming blif_content contains the BLIF file content
if __name__ == "__main__":
    # Read the BLIF content from the provided file
    with open('dec.blif', 'r') as f:
        blif_content = f.read()
    
    # Analyze the circuit
    stats = analyze_blif(blif_content)
    print(f"Circuit Analysis for {stats['model_name']}:")
    print(f"- Inputs: {stats['num_inputs']}")
    print(f"- Outputs: {stats['num_outputs']}")
    print(f"- Gates: {stats['num_gates']}")
    print(f"- Connections: {stats['num_connections']}")
    print(f"- Gate types: {stats['gate_types']}")
    
    # Generate visualizations
    plt1 = visualize_circuit(blif_content)
    plt1.savefig('circuit_graph.png')
    
    plt2 = create_structural_visualization(blif_content)
    plt2.savefig('circuit_structure.png')
    
    plt.show()