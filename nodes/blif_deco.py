class BlifDecoder:
    def __init__(self, blif_file=None):
        """
        Inicializa un nuevo decodificador a partir de un archivo BLIF
        

        Args:
            blif_file (str): Ruta al archivo BLIF a cargar
        """
        # Almacenamiento de la estructura
        self.model_name = ""
        self.inputs = []         # Lista de nombres de entradas
        self.outputs = []        # Lista de nombres de salidas
        self.nodes = {}          # Diccionario de nodos internos
        self.functions = {}      # Funciones lógicas para cada salida/nodo
        
        # Estado actual
        self.input_values = {}   # Valores actuales de las entradas
        self.output_values = {}  # Valores actuales de las salidas
        self.node_values = {}    # Valores actuales de los nodos internos
        
        # Cargar desde archivo si se proporciona
        if blif_file:
            self.load_from_file(blif_file)
    
    def load_from_file(self, blif_file):
        """Carga la estructura desde un archivo BLIF"""
        with open(blif_file, 'r') as f:
            lines = f.readlines()
        
        # Eliminar comentarios y líneas vacías
        lines = [line.split('//')[0].strip() for line in lines if line.strip()]
        
        # Procesar líneas
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Procesar definición del modelo
            if line.startswith('.model'):
                self.model_name = line.split()[1]
            
            # Procesar entradas
            elif line.startswith('.inputs'):
                inputs_str = line[7:].strip() # strip() elimina espacios al inicio y final
                # Manejar continuación de línea con '\'
                while line.endswith('\\') and i+1 < len(lines):
                    i += 1
                    line = lines[i].strip()
                    inputs_str += ' ' + line
                
                # Almacenar nombres de entradas
                input_names = inputs_str.replace('\\', '').split()
                self.inputs = input_names
                
                # Inicializar valores de entrada
                for input_name in self.inputs:
                    self.input_values[input_name] = 0
            
            # Procesar salidas
            elif line.startswith('.outputs'):
                outputs_str = line[8:].strip()
                # Manejar continuación de línea con '\'
                while line.endswith('\\') and i+1 < len(lines):
                    i += 1
                    line = lines[i].strip()
                    outputs_str += ' ' + line
                
                # Almacenar nombres de salidas
                output_names = outputs_str.replace('\\', '').split()
                self.outputs = output_names
                
                # Inicializar valores de salida
                for output_name in self.outputs:
                    self.output_values[output_name] = 0
            
            # Procesar definiciones de nodos
            elif line.startswith('.names'):
                parts = line.split()
                output_node = parts[-1]
                input_nodes = parts[1:-1]
                
                # Leer tabla de verdad
                truth_table = []
                i += 1
                while i < len(lines) and not lines[i].startswith('.'):
                    truth_table.append(lines[i])
                    i += 1
                i -= 1  # Retroceder para no saltarnos la siguiente directiva
                
                # Almacenar definición del nodo
                if output_node.startswith('n'):
                    self.nodes[output_node] = {
                        'inputs': input_nodes,
                        'truth_table': truth_table
                    }
                    self.node_values[output_node] = 0
                
                # Almacenar función lógica
                self.functions[output_node] = {
                    'inputs': input_nodes,
                    'truth_table': truth_table
                }
            
            i += 1
    
    def set_input(self, input_name, value):
        """Establece el valor de una entrada específica"""
        if input_name in self.input_values:
            self.input_values[input_name] = value
        else:
            raise ValueError(f"Entrada no encontrada: {input_name}")
    
    def set_inputs_from_vector(self, vector):
        """Establece los valores de entrada a partir de un vector binario"""
        if len(vector) != len(self.inputs):
            raise ValueError(f"El vector debe tener {len(self.inputs)} bits")
        
        for i, input_name in enumerate(self.inputs):
            self.input_values[input_name] = vector[i]
    
    def get_output(self, output_name):
        """Obtiene el valor de una salida específica"""
        if output_name in self.output_values:
            return self.output_values[output_name]
        else:
            raise ValueError(f"Salida no encontrada: {output_name}")
    
    def get_outputs_as_vector(self):
        """Obtiene un vector con todos los valores de salida"""
        return [self.output_values[output] for output in self.outputs]
    
    def simulate(self):
        """
        Simula el circuito con los valores de entrada actuales
        y actualiza los valores de salida
        """
        # Reiniciar valores de nodos y salidas
        for node in self.node_values:
            self.node_values[node] = 0
            
        for output in self.output_values:
            self.output_values[output] = 0
        
        # Evaluar nodos en orden topológico (suponemos que ya están ordenados en el archivo BLIF)
        for node_name, node_info in self.nodes.items():
            self.node_values[node_name] = self._evaluate_function(node_info)
        
        # Evaluar salidas
        for output_name in self.outputs:
            if output_name in self.functions:
                self.output_values[output_name] = self._evaluate_function(self.functions[output_name])
    
    def _evaluate_function(self, function_info):
        """
        Evalúa una función lógica basada en su tabla de verdad
        
        Args:
            function_info (dict): Información de la función (entradas y tabla de verdad)
            
        Returns:
            int: Resultado de la función (0 o 1)
        """
        # Para simplificar, asumimos que todas las funciones son del tipo "AND"
        # donde cada fila de la tabla de verdad indica una combinación que da 1
        for row in function_info['truth_table']:
            parts = row.split()
            if len(parts) >= 2 and parts[-1] == "1":
                input_pattern = parts[0]
                match = True
                
                for i, input_name in enumerate(function_info['inputs']):
                    expected_value = None
                    if i < len(input_pattern):
                        if input_pattern[i] == '0':
                            expected_value = 0
                        elif input_pattern[i] == '1':
                            expected_value = 1
                    
                    if expected_value is not None:
                        actual_value = self._get_value(input_name)
                        if actual_value != expected_value:
                            match = False
                            break
                
                if match:
                    return 1
        
        return 0
    
    def _get_value(self, name):
        """Obtiene el valor actual de un nodo o entrada"""
        if name in self.input_values:
            return self.input_values[name]
        elif name in self.node_values:
            return self.node_values[name]
        else:
            raise ValueError(f"Nodo o entrada no encontrado: {name}")
    
    def print_status(self):
        """Imprime el estado actual del circuito"""
        print("--- Estado del Circuito ---")
        print("Entradas:")
        for input_name in self.inputs:
            print(f"  {input_name}: {self.input_values[input_name]}")
        
        print("\nNodos internos:")
        for node_name in sorted(self.nodes.keys()):
            print(f"  {node_name}: {self.node_values[node_name]}")
        
        print("\nSalidas:")
        for output_name in self.outputs:  # Mostrar solo las primeras 10 salidas
            print(f"  {output_name}: {self.output_values[output_name]}")
        # print(f"  ... y {len(self.outputs) - 10} más")

    def get_specific_output(self, input_vector, output_name):
        """
        Simula el circuito con el vector de entrada especificado y devuelve
        el valor de una única salida.
        
        Args:
            input_vector (list): Vector con los valores de entrada
            output_name (str): Nombre de la salida deseada (ej: "selectp1[0]")
            
        Returns:
            int: Valor de la salida especificada (0 o 1)
        """
        self.set_inputs_from_vector(input_vector)
        self.simulate()
        return self.get_output(output_name)

    def get_active_output(self, input_vector):
        """
        Simula el circuito con el vector de entrada especificado y devuelve
        la posición de la salida que se activó (valor 1).
        
        Args:
            input_vector (list): Vector con los valores de entrada
        
        Returns:
            dict: Diccionario con {'output_name': nombre_salida, 'output_index': índice}
                 de la salida que se activó
        """
        self.set_inputs_from_vector(input_vector)
        self.simulate()
        
        # Buscar qué salida se activó
        for output_name in self.output_values:
            if self.output_values[output_name] == 1:
                # Extraer el tipo e índice (ej: selectp1[42] -> 'selectp1', 42)
                if '[' in output_name:
                    name_part = output_name.split('[')[0]
                    index_part = int(output_name.split('[')[1].split(']')[0])
                    return {'output_name': output_name, 'output_type': name_part, 'output_index': index_part}
                else:
                    return {'output_name': output_name, 'output_index': -1}
        
        return None  # Si ninguna salida está activa

# Ejemplo de uso:
if __name__ == "__main__":
    # Crear una instancia del decodificador
    decoder = BlifDecoder("dec.blif")
    
    # Establecer valores de entrada
    test_vector = [0, 0, 0, 0, 0, 0, 0, 1]  # Vector de prueba
    decoder.set_inputs_from_vector(test_vector)
    
    # Simular el circuito
    decoder.simulate()
    
    # Mostrar resultados
    decoder.print_status()
    
    # Obtener un vector de salida específico
    selectp1_outputs = [decoder.get_output(f"selectp1[{i}]") for i in range(128)]
    selectp2_outputs = [decoder.get_output(f"selectp2[{i}]") for i in range(128)]
    result = decoder.get_active_output(test_vector)
    print(f"Para la entrada {test_vector}, se activó: {result['output_name']} (índice: {result['output_index']})")
    # print("\nPrimeros 10 bits de selectp1:", selectp1_outputs[:10])