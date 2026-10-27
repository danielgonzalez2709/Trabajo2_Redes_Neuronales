# %% [markdown]
# # B9 — pruebas de `src/pd/espacio.py` (constantes del espacio de búsqueda, B-pd §3.3)
#
# Las constantes viven en un solo lugar: `src/pd/hpo.py` y `src/pd/tablas.py` las importan de `espacio.py`, que no
# depende de Optuna ni de torch (en Jupyter el import de Optuna imprime un aviso de tqdm con la ruta local).

# %%
import subprocess  # procesos limpios para revisar qué módulos se importan
import sys  # intérprete actual
from pathlib import Path  # raíz del repositorio

import pytest  # parametrización

RAIZ = Path(__file__).resolve().parents[1]  # raíz del repositorio
NOMBRES = ("CLAVES_ESPACIO", "RANGOS", "CLAVES_BUSCADAS", "ESTADO_COMPLETO")  # constantes compartidas


# %% [markdown]
# ## Módulos que arrastra un import
#
# Ejecuta `import <modulo>` en un proceso nuevo (este ya pudo importar Optuna) y devuelve qué módulos pesados quedaron
# cargados.

# %%
def _cargados(modulo: str) -> dict:
    """{'optuna', 'torch', 'tqdm.auto'} → ¿quedaron en sys.modules tras importar ``modulo``?"""
    codigo = (f"import sys, {modulo}; "  # import a revisar
              "print(*[m in sys.modules for m in ('optuna', 'torch', 'tqdm.auto')])")  # tres booleanos
    salida = subprocess.run([sys.executable, "-c", codigo], cwd=RAIZ, capture_output=True, text=True,  # ejecuta
                            check=True).stdout.split()  # ['False', 'False', 'False']
    return dict(zip(("optuna", "torch", "tqdm.auto"), (s == "True" for s in salida)))  # nombre → cargado


# %% [markdown]
# ## `espacio.py` es ligero

# %%
def test_espacio_no_importa_optuna_ni_torch():
    """Importar src.pd.espacio no carga Optuna, torch ni tqdm.auto."""
    assert _cargados("src.pd.espacio") == {"optuna": False, "torch": False, "tqdm.auto": False}  # ligero


# %% [markdown]
# ## `tablas.py` no arrastra Optuna

# %%
def test_tablas_no_importa_optuna():
    """Importar src.pd.tablas no carga Optuna ni tqdm.auto (el que avisa en Jupyter)."""
    cargados = _cargados("src.pd.tablas")  # tablas importa experimento (con torch), no hpo
    assert not cargados["optuna"] and not cargados["tqdm.auto"]  # sin Optuna


# %% [markdown]
# ## Una sola fuente: `hpo.py` y `tablas.py` re-exportan los mismos objetos

# %%
@pytest.mark.parametrize("nombre", NOMBRES)  # cada constante
def test_hpo_y_tablas_usan_las_constantes_de_espacio(nombre):
    """hpo.<nombre> y tablas.<nombre> son el mismo objeto que espacio.<nombre> (no una copia)."""
    from src.pd import espacio, hpo, tablas  # los tres módulos
    assert getattr(hpo, nombre) is getattr(espacio, nombre)  # hpo re-exporta
    assert getattr(tablas, nombre) is getattr(espacio, nombre)  # tablas importa


# %% [markdown]
# ## Valores (B-pd §3.3) y el estado de Optuna

# %%
def test_valores_del_espacio():
    """Las 8 claves del espacio, la escala de los rangos y los parámetros buscados, en su orden."""
    from src.pd import espacio  # módulo bajo prueba
    assert espacio.CLAVES_ESPACIO == ("n_capas", "neuronas", "activacion", "dropout", "l2", "optimizador", "lr",
                                      "batch")  # B-pd §3.3
    assert espacio.RANGOS == {"n_capas": "entero", "dropout": "lineal", "l2": "log", "lr": "log"}  # escalas
    assert espacio.CLAVES_BUSCADAS == ("capas", "activacion", "dropout", "l2", "optimizador", "lr", "batch")  # orden


# %% [markdown]
# El texto del estado se fija sin importar Optuna; esta prueba lo ata al nombre real del estado.

# %%
def test_estado_completo_es_el_de_optuna():
    """El texto 'COMPLETE' coincide con el nombre del estado de Optuna (aquí sí se importa Optuna)."""
    import optuna  # solo en la prueba
    from src.pd import espacio  # módulo bajo prueba
    assert espacio.ESTADO_COMPLETO == optuna.trial.TrialState.COMPLETE.name  # mismo texto
