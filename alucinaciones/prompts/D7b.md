# D7b — Blog 08 con los resultados de D4b (José, 26-oct)

Prompt literal enviado al subagente `redactor-tecnico`:

---

Tarea D7b de la Parte D (LGD), rama `dev/jose`. Repo: `C:\Users\Jose Miguel\Documents\ia\Trabajo2_Redes_Neuronales`.
Actualiza `blog/secciones/08-lgd.md` a los resultados regenerados en D4b (commit `d8b2911`). Sin commit. Solo tocas ese
archivo (y `blog/references.bib` solo si hace falta una referencia que ya esté verificada en el repo; nunca inventes citas:
`TODO(cita)`).

## Qué cambió (fuente de verdad: los JSON, nunca la memoria)
- L2 = (λ/2)‖W‖² ≡ `weight_decay = λ`, sin sesgos (B-pd §3.2, v0.13). El blog hoy dice λΣ‖W‖² y `weight_decay = 2λ`
  (líneas ~202-203) y "L2 0,0033" (línea ~217): corrígelo. Menciona en una frase que antes D usaba λ‖W‖² y que se alineó con B
  para que el mismo rango del espacio de búsqueda signifique lo mismo en las dos partes (comparación justa).
- Semilla representativa: MAE de **validación** mediano (posición (n−1)//2 por (mae_val, seed)); antes se elegía por test.
  Explica en una frase por qué (test no decide).
- `summary.json` ahora es por modelo: `lineal`, `mlp`, `referencias` en la raíz (antes `modelos.*`). Revisa cualquier ruta de
  campo citada en el texto.
- Todas las cifras (MAE, RMSE val/test con media ± desv, mejor/peor; lineal; referencias constantes; n_parametros; mejor
  ensayo de `results/data/lgd/hpo_ensayos.json` con su configuración; presupuesto: n_configuraciones_busqueda,
  tiempo_busqueda_s, tiempo_semillas_s, tiempo_total_s; semilla mediana) se leen de `results/data/lgd/summary.json`,
  `hpo_ensayos.json` y `semillas/`. Escribe un script en el scratchpad que las imprima y úsalo; no copies números del texto viejo.
- Hallazgos que el blog debe reportar con su evidencia: (a) la mediana constante de train sigue ganando en MAE y RMSE de test;
  (b) superficie de búsqueda casi plana (cuántos ensayos quedan a < 0,001 de la mejor pérdida de val: calcúlalo); (c) el MLP
  predice en un rango estrecho (mín y máx de `pred_mlp` en `results/data/lgd/pred_test.parquet`), cercano a una constante;
  (d) en val la lineal gana al MLP y en test el MLP gana a la lineal (cuántas semillas de 30, calcúlalo). Discute (a)–(d) con la
  censura y la masa en LGD = 1 que el blog ya documenta, sin afirmar causas no probadas.
- Las figuras `lgd-*.png` ya están regeneradas; verifica que los pies de figura sigan siendo ciertos (p. ej. la etiqueta de
  `lgd-semillas.png` ahora dice "Semilla mediana por MAE de val").

## Reglas
Español neutro y profesional, como el resto del blog. Fórmulas en LaTeX como ya están. Mantén la estructura de secciones.
No toques otras secciones del blog (la 09 es de Daniel). Reporta: lista de cambios por sección, el script usado y su salida,
y cualquier afirmación del texto viejo que no pudiste verificar.
