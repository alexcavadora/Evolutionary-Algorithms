def generar_mascara_dama(fila, columna):
    # Máscara para la fila
    mascara_fila = 0xFF << (fila * 8)  # 0xFF es 11111111 en binario

    # Máscara para la columna
    mascara_columna = 0x0101010101010101 << columna

    # Máscara para la diagonal principal (de izquierda a derecha)
    mascara_diagonal_principal = 0
    f, c = fila, columna
    while f >= 0 and c >= 0:
        mascara_diagonal_principal |= 1 << (f * 8 + c)
        f -= 1
        c -= 1
    f, c = fila + 1, columna + 1
    while f < 8 and c < 8:
        mascara_diagonal_principal |= 1 << (f * 8 + c)
        f += 1
        c += 1

    # Máscara para la diagonal secundaria (de derecha a izquierda)
    mascara_diagonal_secundaria = 0
    f, c = fila, columna
    while f >= 0 and c < 8:
        mascara_diagonal_secundaria |= 1 << (f * 8 + c)
        f -= 1
        c += 1
    f, c = fila + 1, columna - 1
    while f < 8 and c >= 0:
        mascara_diagonal_secundaria |= 1 << (f * 8 + c)
        f += 1
        c -= 1

    # Combinar todas las máscaras
    mascara_total = mascara_fila | mascara_columna | mascara_diagonal_principal | mascara_diagonal_secundaria

    return mascara_total