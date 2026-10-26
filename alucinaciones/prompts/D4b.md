# D4b — Rehacer D4 con L2 (λ/2)‖W‖², `evidencia` y `summary.json` como B (José, 26-oct)

Prompt literal enviado al subagente `analista-experimentos`:

---

Tarea D4b de la Parte D (LGD), rama `dev/jose`. Repo: `C:\Users\Jose Miguel\Documents\ia\Trabajo2_Redes_Neuronales`.
Python: `.venv/Scripts/python.exe` (Windows, CPU). Sin internet. Sin commit (lo hace el orquestador).

## Por qué
D4 (25-oct) se corrió con L2 = l2·Σ‖W‖² (el doble de penalización que B con el mismo número) y sin `evidencia`. D9/D10
(commits `0a5fcf0`…`3b40104` y el siguiente) cambiaron: L2 = (λ/2)‖W‖² ≡ `weight_decay = λ` (como B), huellas §6,
`summary.json` por modelo como B y semilla representativa por MAE de **val** mediano. Spec: `docs/spec/partes/D-lgd.md`
§3.3–§3.4; núcleo `docs/spec/SPEC.md` §5.4 y §6 (comparación justa: mismo preprocesamiento y partición, presupuesto
explícito, 30 semillas, parámetros).

## Pasos
0. Copia al scratchpad de tu sesión (o a una carpeta temporal fuera del repo) `results/data/lgd/{summary.json,hpo_ensayos.json}`
   y `configs/lgd/mlp_final.yaml` actuales, para comparar.
1. `.venv/Scripts/python.exe run.py lgd-hpo --config configs/lgd/hpo.yaml` (30 ensayos, datos completos, sin `--set`).
   Reescribe `configs/lgd/mlp_final.yaml` y `results/data/lgd/hpo_ensayos.json`. Mide el tiempo.
2. `.venv/Scripts/python.exe run.py lgd --config configs/lgd/base.yaml` (lineal + 30 semillas). Escribe `summary.json`,
   `semillas/`, `pred_test.parquet`, `pred_pd_test.parquet`, `models/lgd_lineal.joblib`, `models/lgd_mlp.pt`.
   Verifica que `mlp.config_de_busqueda` es true y que `evidencia` es idéntica en summary, semillas y hpo.
3. Reproducibilidad: vuelve a correr el paso 2 y confirma que métricas y predicciones quedan idénticas (o reporta la
   diferencia máxima).
4. `.venv/Scripts/python.exe run.py figures`: regenera todas las figuras, pero **solo deben quedar cambios en
   `blog/docs/figures/lgd-*.png`**. Cualquier otra figura modificada se restaura con `git checkout -- <ruta>` (no son de D).
5. Notebook 04: regenéralo solo a él (mira `src/common/cli.py`/`run.py notebooks` para ver si acepta un nombre; si solo
   regenera todos, restaura con `git checkout` los `.ipynb` que no sean `notebooks/04_lgd.ipynb`). Debe correr sin error.
6. Corre `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --ignore=tests/test_hpo.py` (test_hpo es de B y falla en
   Windows por permisos) y `-m lento tests/test_demo_lgd.py`. Las pruebas con `TODO(retirar-tras-D4b)` ya deberían correr
   con los resultados nuevos: repórtalo (no las retires tú; lo decide el orquestador).
7. `git status --short`: lista exacta de archivos cambiados.

## Reporte (cifras leídas de los JSON, nunca de memoria)
- Mejor ensayo: número, parámetros (capas, activación, dropout, l2, optimizador, lr, batch), pérdida de val, n_parametros;
  `en_borde` si existe. Comparación con el mejor anterior (ensayo 27, l2 0,00334 en la convención vieja = λ 0,00668).
- Tabla test y val: MAE y RMSE (media ± desv, mejor, peor) del MLP, la lineal y las referencias constantes (media y mediana
  de train). ¿Sigue ganando la mediana constante en MAE? ¿Cambió el ranking frente a D4?
- Semilla representativa nueva vs anterior (10). Presupuesto: `n_configuraciones_busqueda`, `tiempo_busqueda_s`,
  `tiempo_semillas_s`, `tiempo_total_s`.
- Comparación justa: confirma las 30 semillas, la misma evidencia en todos los JSON y que el espacio de búsqueda es el de B
  (`configs/pd/hpo.yaml` vs `configs/lgd/base.yaml`).
- Pytest literal y la lista de archivos cambiados. Cualquier anomalía (divergencias, ensayos FAIL, `en_borde`).
