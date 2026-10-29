"""
Actividad 2.3 — Búsqueda local
Hill Climbing y Simulated Annealing
Requisitos: Python 3; Matplotlib para la gráfica (pip install matplotlib).
"""
import math
import random
import time
import matplotlib.pyplot as plt

def funcion_objetivo(x):
    """Función que queremos maximizar."""
    return x * math.sin(10 * math.pi * x) + 1

def hill_climbing(x_inicial, iteraciones=2000, paso=0.05, minimo=-1, maximo=2):
    """
    Examina dos vecinos (x-paso y x+paso) y avanza al mejor
    solamente si mejora el resultado.
    """
    x = x_inicial
    mejor_valor = funcion_objetivo(x)
    historial = []

    for _ in range(iteraciones):
        vecinos = [
            max(minimo, min(maximo, x - paso)),
            max(minimo, min(maximo, x + paso))
        ]
        mejor_vecino = max(vecinos, key=funcion_objetivo)

        if funcion_objetivo(mejor_vecino) > mejor_valor:
            x = mejor_vecino
            mejor_valor = funcion_objetivo(x)

        historial.append(mejor_valor)

    return x, mejor_valor, historial

def simulated_annealing(x_inicial, iteraciones=2000, paso=0.05,
                        temperatura_inicial=100, enfriamiento=0.995,
                        minimo=-1, maximo=2, semilla=42):
    """
    Prueba un vecino aleatorio. Acepta mejoras siempre y a veces
    acepta empeoramientos según la temperatura.
    """
    rng = random.Random(semilla)
    x = x_inicial
    valor_actual = funcion_objetivo(x)
    mejor_x = x
    mejor_valor = valor_actual
    temperatura = temperatura_inicial
    historial = []

    for _ in range(iteraciones):
        vecino = max(minimo, min(maximo, x + rng.uniform(-paso, paso)))
        valor_vecino = funcion_objetivo(vecino)
        diferencia = valor_vecino - valor_actual

        if diferencia > 0 or rng.random() < math.exp(
            diferencia / max(temperatura, 1e-12)
        ):
            x = vecino
            valor_actual = valor_vecino

        if valor_actual > mejor_valor:
            mejor_x = x
            mejor_valor = valor_actual

        historial.append(mejor_valor)
        temperatura *= enfriamiento

    return mejor_x, mejor_valor, historial

def main():
    # Parámetros del experimento
    x_inicial = 0.15
    iteraciones = 2000
    paso = 0.05
    temperatura_inicial = 100
    enfriamiento = 0.995
    semilla = 42

    inicio = time.perf_counter()
    x_hc, valor_hc, historial_hc = hill_climbing(
        x_inicial, iteraciones, paso
    )
    tiempo_hc = time.perf_counter() - inicio

    inicio = time.perf_counter()
    x_sa, valor_sa, historial_sa = simulated_annealing(
        x_inicial, iteraciones, paso,
        temperatura_inicial, enfriamiento, semilla=semilla
    )
    tiempo_sa = time.perf_counter() - inicio

    print("\\n=== Resultados de búsqueda local ===")
    print(f"Función: f(x) = x * sin(10*pi*x) + 1")
    print(f"Intervalo: [-1, 2] | Inicio: x = {x_inicial}")
    print(f"Iteraciones: {iteraciones} | Paso: {paso}")
    print(f"Temperatura inicial: {temperatura_inicial}")
    print(f"Factor de enfriamiento: {enfriamiento}")
    print(f"Semilla aleatoria: {semilla}\\n")
    print("Hill Climbing:")
    print(f"  Mejor x: {x_hc:.6f}")
    print(f"  Mejor f(x): {valor_hc:.6f}")
    print(f"  Tiempo: {tiempo_hc:.6f} segundos")
    print("\\nSimulated Annealing:")
    print(f"  Mejor x: {x_sa:.6f}")
    print(f"  Mejor f(x): {valor_sa:.6f}")
    print(f"  Tiempo: {tiempo_sa:.6f} segundos")

    plt.figure(figsize=(10, 5.5))
    plt.plot(range(1, iteraciones + 1), historial_hc, label="Hill Climbing")
    plt.plot(range(1, iteraciones + 1), historial_sa, label="Simulated Annealing")
    plt.xlabel("Iteración")
    plt.ylabel("Mejor valor de f(x) encontrado")
    plt.title("Convergencia de los algoritmos de búsqueda local")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("grafica_convergencia_2_3.png", dpi=180)
    print("\\nGráfica guardada como: grafica_convergencia_2_3.png")
    plt.show()

if __name__ == "__main__":
    main()
