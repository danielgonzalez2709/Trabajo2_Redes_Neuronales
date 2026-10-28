# Trabajo 2 — Redes neuronales para datos tabulares (¿me prestas?)

Riesgo de crédito con Lending Club: probabilidad de incumplimiento (PD), pérdida dado incumplimiento (LGD), scorecard,
app web y video. Curso Redes neuronales y algoritmos bioinspirados, UNAL 2026-02.

- Spec (fuente de verdad, con el enunciado en el Anexo): [`docs/spec/SPEC.md`](docs/spec/SPEC.md) · partes en
  [`docs/spec/partes/`](docs/spec/partes/) · tablero: [`docs/spec/PLAN.md`](docs/spec/PLAN.md)
- Estado actual: [`docs/MEMORIA.md`](docs/MEMORIA.md)
- Reporte (blog): https://danielgonzalez2709.github.io/Trabajo2_Redes_Neuronales/
- Aplicación web: https://trabajo2-redes-neuronales.vercel.app (los dos modelos de PD, con selector)

## Instalación (Python 3.12)

Windows:

```bash
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Linux y macOS: en Linux, la rueda de torch de PyPI trae CUDA (unos 3 GB), así que primero se instala la de CPU desde el
índice de PyTorch y después el resto (es lo mismo que hace el CI en `.github/workflows/ci.yml`):

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
```
En macOS la rueda de PyPI ya es de CPU; el primer comando de pip se puede omitir.

Con GPU NVIDIA (opcional): reinstalar torch desde https://pytorch.org con la versión de CUDA de su equipo. La demo corre en CPU.

## Datos
Los datos procesados ya están en `data/processed/` (no hace falta descargar nada para correr los modelos).
Para reconstruirlos desde el crudo de Kaggle, ver [`data/fuentes.md`](data/fuentes.md) y correr:

```bash
python scripts/preparar_datos.py
```

## Comandos
Ver núcleo §4.1 del Spec. Cada parte conecta su subcomando de `run.py` (`python run.py --help`).

## Aplicación web en local (Node.js)

```bash
npm --prefix app ci
npm --prefix app run dev
```
Abre http://localhost:3000. La versión publicada se despliega en Vercel desde `main` (*Root Directory* = `app`).

## Pruebas

```bash
python -m pytest -q
```

## Semillas
Semilla efectiva = `base_seed + run_id` (`src/common/seeds.py`); `base_seed` está en cada archivo de `configs/`.
