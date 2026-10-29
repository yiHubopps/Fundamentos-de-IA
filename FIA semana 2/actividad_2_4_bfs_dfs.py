
from collections import deque
import time

# LABERINTO:
# S = inicio, G = meta, 0 = camino libre, 1 = pared

laberinto = [
    ['S', 0, 1, 0, 0, 0, 1],
    [ 1, 0, 1, 0, 1, 0, 1],
    [ 0, 0, 0, 0, 1, 0, 0],
    [ 0, 1, 1, 0, 0, 0, 1],
    [ 0, 0, 0, 1, 1, 0, 0],
    [ 1, 1, 0, 0, 0, 1, 0],
    [ 0, 0, 0, 1, 0, 0, 'G']
]

FILAS = len(laberinto)
COLUMNAS = len(laberinto[0])

INICIO = (0, 0)
META = (6, 6)

# Arriba, abajo, izquierda y derecha
movimientos = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1)
]


def obtener_vecinos(posicion):
    fila, columna = posicion
    vecinos = []

    for df, dc in movimientos:
        nf = fila + df
        nc = columna + dc

        if 0 <= nf < FILAS and 0 <= nc < COLUMNAS:
            if laberinto[nf][nc] != 1:
                vecinos.append((nf, nc))

    return vecinos


def reconstruir_camino(padres, meta):
    camino = []
    actual = meta

    while actual is not None:
        camino.append(actual)
        actual = padres[actual]

    camino.reverse()
    return camino


def bfs(inicio, meta):
    # BFS utiliza una cola FIFO
    cola = deque([inicio])
    visitados = {inicio}
    padres = {inicio: None}
    expandidos = 0

    while cola:
        actual = cola.popleft()
        expandidos += 1

        if actual == meta:
            camino = reconstruir_camino(padres, meta)
            return camino, expandidos, len(visitados)

        for vecino in obtener_vecinos(actual):
            if vecino not in visitados:
                visitados.add(vecino)
                padres[vecino] = actual
                cola.append(vecino)

    return None, expandidos, len(visitados)


def dfs(inicio, meta):
    # DFS utiliza una pila LIFO
    pila = [inicio]
    visitados = {inicio}
    padres = {inicio: None}
    expandidos = 0

    while pila:
        actual = pila.pop()
        expandidos += 1

        if actual == meta:
            camino = reconstruir_camino(padres, meta)
            return camino, expandidos, len(visitados)

        for vecino in reversed(obtener_vecinos(actual)):
            if vecino not in visitados:
                visitados.add(vecino)
                padres[vecino] = actual
                pila.append(vecino)

    return None, expandidos, len(visitados)


def mostrar_resultados(nombre, resultado):
    camino, expandidos, descubiertos = resultado

    print("\n", nombre)
    if camino is None:
        print("No se encontró una solución.")
        return

    print("Camino:", " -> ".join(map(str, camino)))
    print("Pasos:", len(camino) - 1)
    print("Nodos expandidos:", expandidos)
    print("Nodos descubiertos:", descubiertos)


def medir_tiempo(algoritmo, repeticiones=2000):
    inicio = time.perf_counter()

    for _ in range(repeticiones):
        algoritmo(INICIO, META)

    tiempo = (time.perf_counter() - inicio) / repeticiones
    return tiempo


def main():
    print("ACTIVIDAD 2.4: BFS Y DFS")
    print("S = inicio | G = meta")
    print("0 = camino libre | 1 = pared")

    resultado_bfs = bfs(INICIO, META)
    resultado_dfs = dfs(INICIO, META)

    mostrar_resultados("BFS - Busqueda en anchura", resultado_bfs)
    mostrar_resultados("DFS - Busqueda en profundidad", resultado_dfs)

    tiempo_bfs = medir_tiempo(bfs)
    tiempo_dfs = medir_tiempo(dfs)

    print("\nCOMPARACION")
    print(f"Tiempo promedio BFS: {tiempo_bfs:.8f} segundos")
    print(f"Tiempo promedio DFS: {tiempo_dfs:.8f} segundos")

    print("\nCONCLUSION")
    print("BFS encuentra el camino con menos pasos cuando")
    print("todos los movimientos tienen el mismo costo.")
    print("DFS explora en profundidad, pero no garantiza")
    print("encontrar el camino mas corto.")


if __name__ == "__main__":
    main()
