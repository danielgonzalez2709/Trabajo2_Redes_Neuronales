# Trabajo 2 — Redes neuronales para datos tabulares (¿me prestas?)

Riesgo de crédito con Lending Club: probabilidad de incumplimiento (PD), pérdida dado incumplimiento (LGD), scorecard,
app web y video. Curso Redes neuronales y algoritmos bioinspirados, UNAL 2026-02.

- Spec (fuente de verdad, con el enunciado en el Anexo): [`docs/spec/SPEC.md`](docs/spec/SPEC.md) · partes en
  [`docs/spec/partes/`](docs/spec/partes/) · tablero: [`docs/spec/PLAN.md`](docs/spec/PLAN.md)
- Estado actual: [`docs/MEMORIA.md`](docs/MEMORIA.md)

## Instalación (Python 3.12)

```bash
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```
Con GPU NVIDIA (opcional): reinstalar torch desde https://pytorch.org con la versión de CUDA de su equipo. La demo corre en CPU.

## Datos
Los datos procesados ya están en `data/processed/` (no hace falta descargar nada para correr los modelos).
Para reconstruirlos desde el crudo de Kaggle, ver [`data/fuentes.md`](data/fuentes.md) y correr:

```bash
python scripts/preparar_datos.py
```

## Comandos
Ver núcleo §4.1 del Spec. Cada parte conecta su subcomando de `run.py` (`python run.py --help`).

## Pruebas

```bash
python -m pytest -q
```

## Semillas
Semilla efectiva = `base_seed + run_id` (`src/common/seeds.py`); `base_seed` está en cada archivo de `configs/`.
