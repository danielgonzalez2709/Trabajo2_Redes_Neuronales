"""H-07: @dataclass(frozen=True) no congela los campos mutables. Correr desde la raíz del repositorio."""
from dataclasses import FrozenInstanceError, dataclass


@dataclass(frozen=True)
class ResultadoEntrenamiento:  # misma forma que el resultado de src/pd/entrenar.py: un campo historial de tipo list
    historial: list


r = ResultadoEntrenamiento(historial=[0.7, 0.6])
r.historial.append(0.5)  # la lista sí se modifica
print("después de append:", r.historial)
try:
    r.historial = []  # reasignar el campo sí está prohibido
except FrozenInstanceError as e:
    print("reasignar el campo:", type(e).__name__)
