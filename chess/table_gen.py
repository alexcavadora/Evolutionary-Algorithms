import numpy as np

class table:
    def __init__(self,n):
        self.table = self.generate_nxn_bits(n)
        self.n = n

    def copy(self):
        """
        Crea una copia profunda del tablero actual
        Returns:
            table: Una nueva instancia de table con los mismos datos
        """
        new_table = table(self.n)
        new_table.table = self.table.copy()  # NumPy array tiene su propio método copy()
        return new_table
    
    @staticmethod
    def generate_nxn_bits(n):
        """
        Generate array of uint64 to store n*n boolean variables initialized to 0
        Args:
            n: The dimension. Total bits will be n*n
        Returns:
            numpy array of uint64 initialized to 0
        """
        n_bits = n * n
        n_integers = (n_bits + 63) // 64
        return np.zeros(n_integers, dtype=np.uint64)

    # Generar una tabla con n reinas aleatorias
    def generate_random_table(self):
        n = self.n
        tablero = self.generate_nxn_bits(n)
        col_pos = np.random.choice(n, n, replace=False)
        for row in range(n):
            col = col_pos[row]
            reina = row * n + col
            indice = reina // 64
            desplazamiento = reina % 64

            current_value = int(tablero[indice].item())
            shift_value = 1 << int(desplazamiento)
            nuevo_valor = np.uint64(current_value | shift_value)

            tablero[indice] = nuevo_valor
        self.table = tablero

    # Imprimir la tabla en la consola
    def print_table(self):
        n = self.n
        table_data = self.table
        print("   ", end="")
        for col_idx in range(n):
            print(col_idx, end=" ")
        print()

        for row_idx in range(n):
            print(f"{row_idx}  ", end="")
            for col_idx in range(n):
                indice = (row_idx * n + col_idx) // 64
                desplazamiento = (row_idx * n + col_idx) % 64
                if table_data[indice].item() & (1 << desplazamiento):
                    print('R', end=' ')
                else:
                    print('-', end=' ')
            print()

    # Dibujar la tabla en una grafica
    def plot_chessboard(self):
        import matplotlib.pyplot as plt

        # Create a checkerboard pattern
        colors = np.zeros((self.n, self.n))
        for row in range(self.n):
            for col in range(self.n):
                if (row + col) % 2 == 0:
                    colors[row, col] = 1

        # Create a figure and axis
        fig, ax = plt.subplots()
        ax.imshow(colors, cmap='gray', extent=(0, self.n, 0, self.n))

        # Mark queens
        for row in range(self.n):
            for col in range(self.n):
                bit_index = (row * self.n + col)
                idx = bit_index // 64
                offset = bit_index % 64
                # Convert to Python int before bitwise operation
                if int(self.table[idx].item()) & (1 << offset):
                    ax.text(col + 0.5, self.n - row - 0.5, '♛',
                            ha='center', va='center', color='#FFD700', fontsize=14)

        axis = range(self.n)
        ax.set_xticks(axis)
        ax.set_yticks(axis)
        y_label = range(self.n,0,-1)
        ax.set_yticklabels(y_label)

        ax.set_xlim(0, self.n)
        ax.set_ylim(0, self.n)
        ax.invert_yaxis()
        plt.show()

    ## Notas: la tabla impresa en consola y la tabla dibujada en la gráfica no coinciden
    ##        la tabla impresa en consola es correcta, la tabla dibujada en la gráfica no
    ##        se debe a que la gráfica no está tomando en cuenta la inversión de los ejes
    ##        en la gráfica, el eje y se invierte para que las filas se muestren en orden descendente

class mask:
    def __init__(self, n):
        self.n = n
        self.mask_array = [[None for _ in range(n)] for _ in range(n)]  # Inicializar matriz de máscaras
        self.init_mask_array()

    def generate_mask(self, fila, columna):
        """
        Genera una máscara para la dama en (fila, columna).
        La máscara cubre la fila, columna y diagonales, excluyendo la posición de la dama.
        """
        n = self.n
        n_bits = n * n
        n_integers = (n_bits + 63) // 64
        
        # Inicializar la máscara total a 0 en todos los bits
        mascara_total = [0] * n_integers

        def set_bit(arr, pos):
            """Activa el bit en la posición `pos`."""
            idx = pos // 64
            off = int(pos % 64)
            arr[idx] |= (1 << off)

        # Máscara para la fila (excluyendo la posición de la dama)
        for c in range(n):
            if c != columna:
                set_bit(mascara_total, fila * n + c)

        # Máscara para la columna (excluyendo la posición de la dama)
        for r in range(n):
            if r != fila:
                set_bit(mascara_total, r * n + columna)

        # Máscara para la diagonal principal (excluyendo la posición de la dama)
        f, c = fila - 1, columna - 1
        while f >= 0 and c >= 0:
            set_bit(mascara_total, f * n + c)
            f -= 1
            c -= 1
        f, c = fila + 1, columna + 1
        while f < n and c < n:
            set_bit(mascara_total, f * n + c)
            f += 1
            c += 1

        # Máscara para la diagonal secundaria (excluyendo la posición de la dama)
        f, c = fila - 1, columna + 1
        while f >= 0 and c < n:
            set_bit(mascara_total, f * n + c)
            f -= 1
            c += 1
        f, c = fila + 1, columna - 1
        while f < n and c >= 0:
            set_bit(mascara_total, f * n + c)
            f += 1
            c -= 1

        return mascara_total

    def init_mask_array(self):
        """
        Inicializa la matriz de máscaras para todas las posiciones del tablero.
        """
        for i in range(self.n):
            for j in range(self.n):
                self.mask_array[i][j] = self.generate_mask(i, j)

    def apply_mask(self, individual, fila, columna):
        """
        Aplica la máscara a la tabla del individuo.
        """
        table_data = individual.table
        mask_array = self.mask_array[fila][columna]
        for i in range(len(table_data)):
            table_data[i] = np.bitwise_and(table_data[i], np.uint64(mask_array[i]))

if __name__ == '__main__':
    # Create an 8x8 board (smaller size to better visualize the effect)
    from main import GA
    # Parameters
    n = 8           # 8x8 board
    npop = 100      # population size 
    ngen = 1000     # number of generations
    pmut = 0.5     # mutation probability
    pcross = 0.85    # crossover probability
    
    # Create and run GA
    ga = GA(n, npop, None, ngen, pmut, pcross)
    t = table(n)
    
    # Generate random queens
    t.generate_random_table()
    print('Original table with random queens:')
    t.print_table()
    # t.plot_chessboard()
    
    # Create and initialize mask
    m = mask(n)
    m.init_mask_array()
    
    # Apply mask for position (3,3) - this will keep only queens that don't conflict
    # with a queen at position (3,3)
    m.apply_mask(t, 3, 3)
    print('\nTable after applying mask for position (3,3):')
    t.print_table()
    q,c = ga.count_queens_collisions(t)
    print(f"Queens = {q} - Colisiones = {c}")
    t.plot_chessboard()
