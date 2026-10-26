"""H-16: en Optuna 5.0.0, TPESampler() con los valores por defecto es multivariado en un objetivo.

Uso: .venv/bin/python alucinaciones/evidencias/H-16_tpe_multivariado.py > alucinaciones/evidencias/H-16_tpe_multivariado.txt
"""
import inspect  # firma y código fuente del muestreador

import optuna  # versión fijada en requirements.txt

optuna.logging.set_verbosity(optuna.logging.ERROR)  # sin ruido
print("optuna", optuna.__version__)  # versión usada
firma = inspect.signature(optuna.samplers.TPESampler)  # parámetros del constructor
print("multivariate por defecto en la firma:", firma.parameters["multivariate"].default)  # None, no False
fuente = inspect.getsource(optuna.samplers.TPESampler._is_multivariate)  # regla efectiva
print(fuente)  # dice que en un objetivo se usa el multivariado
muestreador = optuna.samplers.TPESampler(seed=0)  # como en src/pd/hpo.py
estudio = optuna.create_study(direction="minimize", sampler=muestreador)  # un objetivo
print("multivariado efectivo:", muestreador._is_multivariate(estudio))  # True
