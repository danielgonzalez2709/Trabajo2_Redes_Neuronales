"""H-18: un reloj falso de paso fijo no detecta trabajo extra dentro de la zona cronometrada."""
import itertools  # contador del reloj falso


def reloj_paso_fijo(paso=1.0):
    """Reloj falso: 0, paso, 2·paso, … (el de la prueba 12 del plan de B5)."""
    contador = itertools.count(start=0.0, step=paso)  # avanza un paso por llamada
    return lambda: next(contador)  # misma firma que time.perf_counter


def cronometrar(reloj, trabajo):
    """Mide ``trabajo`` entre dos lecturas del reloj."""
    inicio = reloj()  # primera lectura
    trabajo()  # lo que se cronometra
    return reloj() - inicio  # diferencia


solo_fit = cronometrar(reloj_paso_fijo(), lambda: None)  # código correcto: solo el fit
fit_y_medir = cronometrar(reloj_paso_fijo(), lambda: sum(range(10**6)))  # mutante: fit + medición dentro
print(f"solo fit: {solo_fit} · fit + medir: {fit_y_medir} · iguales: {solo_fit == fit_y_medir}")  # ambos 1.0
