import itertools
import pickle
from blif_deco import BlifDecoder


class SimulatedCircuit:
    def __init__(self, blif_file):
        self.decoder = BlifDecoder(blif_file)
        self.inputs_history = []  # Almacena vectores de entrada
        self.outputs_history = []  # Almacena nombres de salida activos

        self.simulate()

        
    def generate_all_input_combinations(self):
            num_inputs = len(self.decoder.inputs) # Número de entradas (8)
            all_combinations = list(itertools.product([0, 1], repeat=num_inputs)) # Generar todas las combinaciones posibles
            return all_combinations # shape = (2^num_inputs, num_inputs)
    
    def simulate(self):
        all_combinations = self.generate_all_input_combinations()
        for input_combination in all_combinations:
            result = self.decoder.get_active_output(input_combination)
            output = self.decoder.get_output(result['output_name'])
            
            self.inputs_history.append(tuple(input_combination))
            self.outputs_history.append(output)

        
    
    def get_output_by_vector(self, input_vector):
        # Buscar en el historial
        for idx, stored_input in enumerate(self.inputs_history):
            if stored_input == tuple(input_vector):
                return self.outputs_history[idx]
    
    def compare_outputs(self, output_vector):
        count = 0
        for idx, stored_output in enumerate(self.outputs_history):
            if stored_output != output_vector[idx]:
                count += 1
        return count
        


# Ejemplo de uso
if __name__ == "__main__":
    decoder = SimulatedCircuit("dec.blif")
    
    test_vector = [0, 0, 0, 0, 0, 0, 0, 1]
    
    
    # Obtener salida específica
    output = decoder.get_output_by_vector(test_vector)
    print(f"Para la entrada {test_vector}, se activó: {output}")
    
    # Ver historial almacenado
    print(f"\nEntradas almacenadas: {decoder.inputs_history}")
    print(f"Salidas almacenadas: {decoder.outputs_history}")