"""Paleta única de las figuras (portada del Trabajo 1, T4.1). La usan src/viz/theme.py (Plotly) y src/viz/style.py (matplotlib).

Origen: paleta de referencia validada de la guía de visualización de datos (categórica de 8 tonos en orden fijo,
rampa secuencial azul, tinta y superficie). Validación del 12-oct-2026 (validate_palette.js, superficie #fcfcfb):
los 5 tonos de los métodos de la Parte 1 pasan banda de luminosidad, croma, separación para daltonismo
(ΔE mínimo adyacente 9,1 ≥ 8) y visión normal (19,6 ≥ 15). AVISO de contraste < 3:1 en aqua, amarillo y magenta:
toda figura con esos tonos lleva leyenda y el reporte incluye la tabla con los valores (regla de alivio).

Reglas:
- El color sigue a la ENTIDAD (método, algoritmo), nunca a su posición: COLOR_METODO no se reordena.
- Un solo eje y; texto siempre en tinta (TINTA*), nunca en el color de la serie.
- Las figuras se dibujan siempre sobre SUPERFICIE clara (también en el modo oscuro del blog, dentro de una tarjeta).
"""

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
TINTA_TENUE = "#898781"  # ejes y etiquetas
CUADRICULA = "#e1e0d9"  # línea fina, sólida
EJE = "#c3c2b7"

# Categórica, en el orden fijo de la paleta de referencia.
CATEGORICA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Trabajo 2 — el color sigue a la ENTIDAD: modelos y splits tienen color fijo en todas las figuras.
COLOR_MODELO = {"logistica": CATEGORICA[0], "mlp": CATEGORICA[1], "lineal": CATEGORICA[0]}
NOMBRE_MODELO = {"logistica": "Regresión logística", "mlp": "Perceptrón multicapa", "lineal": "Regresión lineal"}
COLOR_SPLIT = {"train": CATEGORICA[0], "val": CATEGORICA[2], "test": CATEGORICA[1]}
NOMBRE_SPLIT = {"train": "Entrenamiento", "val": "Validación", "test": "Prueba"}
COLOR_TASA = CATEGORICA[7]  # tasa de malos (rojo de la paleta) cuando comparte figura con conteos

# Secuencial (magnitud: curvas de nivel, mapas de calor): un solo tono, claro → oscuro.
SECUENCIAL_AZUL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
                   "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

FUENTES = ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]  # sans del sistema (skill apple-design)
ANCHO_LINEA = 2
TAMANO_MARCADOR = 8
