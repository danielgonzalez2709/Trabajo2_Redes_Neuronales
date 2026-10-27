# Prompts literales — PR de Spec de la Parte C (v0.25; MEMORIA, pendiente 9)

- **Fecha:** 27-oct-2026 · **Orquestador:** sesión de Claude Code de Juan José (Claude Opus 5.5) · **Rama:** `dev/juanjose`
- Tarea solo de documentación: sin implementador. Juan José aprobó la lista de cambios con un ajuste: los tamaños de SHAP y de
  la permutación se escriben como "valores por defecto de `configs/scorecard/mlp.yaml`".

## 1. revisor-spec (literal)

```text
Revisa el PR de Spec v0.25 de la Parte C en C:\Repositorios\Trabajo2_Redes_Neuronales (rama dev/juanjose, cambios sin commit: git diff -- docs/spec/SPEC.md docs/spec/partes/C-scorecard.md docs/MEMORIA.md). Solo lectura; no edites nada. Windows: .venv/Scripts/python. Informe ≤ 15 líneas con veredicto APROBADO o RECHAZADO y hallazgos concretos (archivo:línea).

Es un cambio de documentación para dejar el Spec igual a lo implementado en C1–C8. Comprueba:
1. Regla del núcleo §0.1: versión 0.25 con una línea de historial en la cabecera (qué, por qué, quién y partes afectadas: E). El Anexo (desde "## Anexo") NO cambió (git diff debe estar vacío ahí). No se tocó nada de otras partes (A, B, D, E) en el núcleo ni en otros archivos de docs/spec/partes/.
2. Núcleo §3, fila Interpretabilidad: coincide con src/scorecard/shap_pd.py, configs/scorecard/mlp.yaml y el bloque "metodo" de results/data/scorecard/shap_mlp.json (explainer, masker, salida logit, fondo train, 15 jugadores con las categóricas enteras, explicados de test, n_permutaciones, permutación en test con caída de AUC y aumento de log-loss para los dos modelos). Los tamaños deben presentarse como valores por defecto de mlp.yaml y coincidir con él.
3. Núcleo §4: configs/scorecard/{base,export,mlp}.yaml y src/scorecard/{scorecard,puntos,shap_pd,hipotesis}.py existen; no hay archivos de C en esas carpetas que falten.
4. Núcleo §5.4: cada clave escrita para scorecard.json, scorecard_mlp.json, shap_mlp.json e hipotesis.json existe en los JSON reales de results/data/scorecard/ (claves de primer y segundo nivel, y las de las listas que se nombran) y no falta ninguna clave de primer nivel.
5. C-scorecard.md: §3.1 (600/50/20, float sin redondear) contra configs/scorecard/base.yaml y src/scorecard/scorecard.py; §3.2 (rango p1–p99, categóricas ≥ 1 %, bandas = deciles de train, solo diferencias; tramos del MLP) contra base.yaml, puntos.py y shap_pd.py; §3.3 contra shap_pd.py y blog/references.bib (lundberg2017); §3.4 contra src/scorecard/hipotesis.py y el bloque "criterio" de hipotesis.json (umbral ρ 0,5, z = 2, sensibilidad 0,3/0,7 y sin compuerta, regla confirmada/refutada/no concluyente, logística y PSI sin voto); §5 con los 7 archivos de pruebas reales de la Parte C en tests/; §6 contra alucinaciones/registro.md H-24 (np.isclose del masker, códigos enteros) y H-28 (mezclar bases, no log10). Las tareas C1–C8 y sus estados no cambiaron.
6. docs/MEMORIA.md: se quitó el pendiente 9, hay una línea nueva arriba en la bitácora y el archivo tiene ≤ 150 líneas.
```
- revisor-spec: APROBADO, sin bloqueantes; matiz aplicado (nombrar `max_evals` = 500 junto a las 16 permutaciones).
