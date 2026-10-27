"""H-28 (antes H-27): ¿cambia el factor al usar log10 en vez de ln? Reproducible sin datos: .venv/Scripts/python -I alucinaciones/evidencias/C7_log10.py"""
import math

base, odds_base, pdo = 600.0, 50.0, 20.0


def puntaje(p, log, factor):
    o = (1 - p) / p
    offset = base - factor * log(odds_base)
    return offset + factor * log(o)


def duplicar(log, factor, o=3.0):
    p1, p2 = 1 / (1 + o), 1 / (1 + 2 * o)
    return puntaje(p2, log, factor) - puntaje(p1, log, factor)


# 1) ln consistente: factor = PDO / ln 2
print("ln consistente: duplicar suma", duplicar(math.log, pdo / math.log(2)))
# 2) log10 consistente: factor = PDO / log10 2 (el factor SÍ cambia, el puntaje NO)
f10 = pdo / math.log10(2)
print("log10 consistente: factor", f10, "duplicar suma", duplicar(math.log10, f10))
p = 0.2
print("mismo puntaje en p=0,2:", puntaje(p, math.log, pdo / math.log(2)), puntaje(p, math.log10, f10))
# 3) bases mezcladas: factor con ln, puntaje con log10 (el error real)
print("mezclado: duplicar suma", duplicar(math.log10, pdo / math.log(2)), "esperado", pdo * math.log10(2) / math.log(2))
