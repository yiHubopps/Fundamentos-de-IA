"""
Dos problemas de búsqueda con NetworkX.
Los tiempos son datos didácticos; no son mediciones.
IA:ChatGPT; apoyo en código y explicación.
"""

from collections import deque
from itertools import permutations
from pathlib import Path
import csv
import json
import platform

import matplotlib
matplotlib.use("Agg")  # Permite guardar imágenes sin abrir ventanas.
import matplotlib.pyplot as plt
import networkx as nx

SALIDA = Path(__file__).resolve().parent / "resultados"
LUGARES = {
    "E": "Entrada", "A": "Administración", "B": "Biblioteca",
    "P": "Patio", "C": "Cómputo", "R": "Rampa",
    "T": "Taller", "L": "Laboratorio CADCAM",
}
# Origen -> (destino, minutos). Los sentidos opuestos se declaran aparte.
MAPA = {
    "E": [("A", 3), ("B", 2), ("L", 20)],
    "A": [("P", 4), ("C", 6)],
    "B": [("P", 2), ("R", 5)],
    "P": [("C", 3), ("T", 4), ("A", 4)],
    "R": [("T", 2), ("C", 3)],
    "C": [("L", 4)], "T": [("L", 2)], "L": [],
}


def sucesores_ruta(estado):
    """Genera (nuevo_estado, acción, coste) sin crear todo el grafo."""
    for destino, minutos in MAPA[estado]:
        yield destino, f"Ir de {estado} a {destino}", minutos


# Cada bit indica si ya terminó una tarea; el orden fija su posición.
TAREAS = ("I", "L", "C", "V")
DURACION = {"I": 4, "L": 6, "C": 3, "V": 5}
BIT = {tarea: 1 << i for i, tarea in enumerate(TAREAS)}
PREVIAS = {"I": 0, "L": 0, "C": 0, "V": 7}
# Tiempo adicional para preparar la siguiente tarea; no es simétrico.
PREPARACION = {
    "O": {"I": 2, "L": 4, "C": 1, "V": 0},
    "I": {"I": 0, "L": 1, "C": 2, "V": 3},
    "L": {"I": 3, "L": 0, "C": 4, "V": 2},
    "C": {"I": 3, "L": 3, "C": 0, "V": 1},
    "V": {"I": 0, "L": 0, "C": 0, "V": 0},
}
INICIO_M = (0, "O")  # (máscara de tareas hechas, última tarea).
META_M = (15, "V")


def sucesores_mantenimiento(estado):
    """Solo permite tareas pendientes cuyas precedencias se cumplieron."""
    hechas, ultima = estado
    for tarea in TAREAS:
        pendiente = not (hechas & BIT[tarea])
        habilitada = (hechas & PREVIAS[tarea]) == PREVIAS[tarea]
        if pendiente and habilitada:
            nuevo = (hechas | BIT[tarea], tarea)
            coste = DURACION[tarea] + PREPARACION[ultima][tarea]
            yield nuevo, f"Realizar {tarea}", coste


def construir_grafo(inicio, sucesores):
    """Expansión BFS: materializa estados alcanzables y evita duplicados.

    BFS aquí construye el espacio; Dijkstra busca el coste mínimo después.
    Los estados deben ser hashables: cadenas, tuplas, etcétera.
    """
    grafo = nx.DiGraph()
    grafo.add_node(inicio)
    pendientes = deque([inicio])
    vistos = {inicio}
    while pendientes:
        estado = pendientes.popleft()
        for nuevo, accion, coste in sucesores(estado):
            if coste < 0:
                raise ValueError("Dijkstra requiere pesos no negativos.")
            grafo.add_edge(estado, nuevo, weight=coste, accion=accion)
            if nuevo not in vistos:
                vistos.add(nuevo)
                pendientes.append(nuevo)
    return grafo


def resolver(grafo, inicio, meta):
    """Devuelve coste total y ruta; una meta inaccesible produce error claro."""
    try:
        return nx.single_source_dijkstra(
            grafo, inicio, target=meta, weight="weight"
        )
    except nx.NetworkXNoPath as error:
        raise ValueError("No existe solución con estos datos.") from error


def dibujar(grafo, posiciones, ruta, etiquetas, archivo, titulo):
    """Dibuja flechas, pesos en minutos y la solución en verde."""
    fig, ax = plt.subplots(figsize=(10, 7.2))
    fig.patch.set_facecolor("#F7F9FC")
    ax.set_facecolor("#F7F9FC")
    optimas = set(zip(ruta, ruta[1:]))
    for u, v in grafo.edges:
        curva = 0.0
        if grafo.has_edge(v, u):
            curva = 0.17
        if (u, v) == ("E", "L"):
            curva = -0.60
        color = "#087F70" if (u, v) in optimas else "#9CAFC5"
        nx.draw_networkx_edges(
            grafo, posiciones, edgelist=[(u, v)], ax=ax,
            edge_color=color, width=2.4 if (u, v) in optimas else 1.2,
            arrowsize=18, node_size=1250,
            connectionstyle=f"arc3,rad={curva}",
        )
        nx.draw_networkx_edge_labels(
            grafo, posiciones, edge_labels={(u, v): grafo[u][v]["weight"]},
            ax=ax, font_size=10, rotate=False,
            connectionstyle=f"arc3,rad={curva}",
            bbox={"facecolor": "#F7F9FC", "edgecolor": "none", "pad": 0.8},
        )
    colores = ["#087F70" if n in ruta else "#173651" for n in grafo]
    nx.draw_networkx_nodes(
        grafo, posiciones, node_color=colores, node_size=1250, ax=ax,
        edgecolors="white", linewidths=1.8,
    )
    nx.draw_networkx_labels(
        grafo, posiciones, labels=etiquetas, font_color="white",
        font_size=11, font_weight="bold", ax=ax,
    )
    ax.set_title(titulo, fontsize=16, loc="left", color="#173651", pad=18)
    ax.text(0, -0.06, "Pesos: minutos | Verde: solución óptima",
            transform=ax.transAxes, fontsize=11, color="#42566F")
    ax.margins(0.18)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(archivo, dpi=220, bbox_inches="tight")
    plt.close(fig)


def verificar(grafo_r, coste_r, coste_m):
    """Contrasta con enumeración exhaustiva, solo para instancias pequeñas."""
    rutas = list(nx.all_simple_paths(grafo_r, "E", "L"))
    mejor_r = min(nx.path_weight(grafo_r, r, "weight") for r in rutas)
    assert coste_r == mejor_r == 10
    resultados = []
    # Enumeración independiente de los 3! órdenes previos a V.
    for orden in permutations(("I", "L", "C")):
        secuencia = orden + ("V",)
        anterior, total = "O", 0
        for tarea in secuencia:
            total += PREPARACION[anterior][tarea] + DURACION[tarea]
            anterior = tarea
        resultados.append({"orden": list(secuencia), "minutos": total})
    assert coste_m == min(r["minutos"] for r in resultados) == 25
    return len(rutas), resultados


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    gr = construir_grafo("E", sucesores_ruta)
    gm = construir_grafo(INICIO_M, sucesores_mantenimiento)
    cr, rr = resolver(gr, "E", "L")
    cm, rm = resolver(gm, INICIO_M, META_M)
    cantidad_rutas, ordenes = verificar(gr, cr, cm)
    assert nx.is_directed_acyclic_graph(gm)
    ids = {estado: f"S{i:02d}" for i, estado in enumerate(gm)}
    posiciones_r = {
        "E": (0, 2), "A": (2, 4), "B": (2, 0), "P": (4, 2),
        "R": (4, -1), "C": (6, 4), "T": (6, 0), "L": (8, 2),
    }
    posiciones_m = {}
    for nivel in range(5):
        capa = [s for s in gm if s[0].bit_count() == nivel]
        for i, estado in enumerate(capa):
            posiciones_m[estado] = (2.4 * (i - (len(capa) - 1) / 2), -2 * nivel)
    dibujar(gr, posiciones_r, rr, {n: n for n in gr},
            SALIDA / "grafo_ruta.png", "Traslado en campus: 8 estados")
    dibujar(gm, posiciones_m, rm, ids,
            SALIDA / "grafo_mantenimiento.png", "Mantenimiento: 14 estados")
    nx.write_graphml(gr, SALIDA / "ruta.graphml")
    nx.write_graphml(nx.relabel_nodes(gm, ids), SALIDA / "mantenimiento.graphml")
    with (SALIDA / "estados_mantenimiento.csv").open(
        "w", encoding="utf-8", newline=""
    ) as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["id", "mascara", "hechas", "ultima"])
        for (mascara, ultima), identificador in ids.items():
            hechas = ",".join(t for t in TAREAS if mascara & BIT[t]) or "ninguna"
            escritor.writerow([identificador, mascara, hechas, ultima])
    resultados = {
        "entorno": {"python": platform.python_version(),
                    "networkx": nx.__version__, "matplotlib": matplotlib.__version__},
        "ruta": {"estados": len(gr), "aristas": gr.number_of_edges(),
                 "camino": rr, "minutos": cr, "rutas_verificadas": cantidad_rutas},
        "mantenimiento": {"estados": len(gm), "aristas": gm.number_of_edges(),
                          "camino": [ids[s] for s in rm],
                          "orden": [s[1] for s in rm[1:]], "minutos": cm,
                          "ordenes_verificados": ordenes},
    }
    texto = json.dumps(resultados, ensure_ascii=False, indent=2)
    (SALIDA / "resultados.json").write_text(texto + "\n", encoding="utf-8")
    print(texto)


if __name__ == "__main__":
    main()