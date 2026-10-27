# %% [markdown]
# # Espacio de la búsqueda del MLP de PD (B-pd §3.3): constantes compartidas
#
# Una sola fuente para las claves del espacio, la escala de cada rango, los parámetros que se buscan y el estado de un
# ensayo terminado. Las importan `src/pd/hpo.py` (que las re-exporta) y `src/pd/tablas.py` (notebook 02). Este módulo
# no importa Optuna ni torch: en Jupyter el import de Optuna imprime un aviso de tqdm con la ruta local del entorno.

# %% [markdown]
# ## Constantes
#
# - `CLAVES_ESPACIO`: claves exactas del bloque `hpo.espacio`;
# - `RANGOS`: las claves que son un rango `[mín, máx]` y su escala (las demás son listas de opciones);
# - `CLAVES_BUSCADAS`: claves del bloque `mlp` que arma cada ensayo, en el orden del JSON;
# - `ESTADO_COMPLETO`: nombre del estado de Optuna de un ensayo terminado (`TrialState.COMPLETE.name`; una prueba lo
#   compara con Optuna).

# %%
CLAVES_ESPACIO = ("n_capas", "neuronas", "activacion", "dropout", "l2", "optimizador", "lr", "batch")  # del espacio
RANGOS = {"n_capas": "entero", "dropout": "lineal", "l2": "log", "lr": "log"}  # claves [mín, máx] y su escala
CLAVES_BUSCADAS = ("capas", "activacion", "dropout", "l2", "optimizador", "lr", "batch")  # claves mlp que se buscan
ESTADO_COMPLETO = "COMPLETE"  # = optuna.trial.TrialState.COMPLETE.name
