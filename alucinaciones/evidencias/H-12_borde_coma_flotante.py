"""H-12: el caso de prueba dropout = 0.475 → "max" (f = 5 % de [0, 0.5]) es falso en coma flotante.

Uso: .venv/bin/python alucinaciones/evidencias/H-12_borde_coma_flotante.py > alucinaciones/evidencias/H-12_borde_coma_flotante.txt
"""
ancho = 0.5 - 0.0  # rango de dropout en configs/pd/hpo.yaml
tolerancia = 0.05 * ancho  # FRACCION_BORDE · ancho
distancia = 0.5 - 0.475  # distancia al máximo del caso sugerido
print("distancia =", repr(distancia), "· tolerancia =", repr(tolerancia))  # 0.025000000000000022 vs 0.025
print("¿en el borde (distancia <= tolerancia)?", distancia <= tolerancia)  # False
print("con valores exactos en binario (f = 0.25):", 0.5 - 0.375 <= 0.25 * 0.5)  # True: 0.125 <= 0.125
