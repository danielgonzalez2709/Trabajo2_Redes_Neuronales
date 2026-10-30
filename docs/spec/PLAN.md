# PLAN — tablero global (Spec v0.27)

> Una línea por tarea. El detalle (aceptación, verificador, especificación) está en la spec de cada parte
> (`docs/spec/partes/`). Pipeline por tarea: **implementador → revisor-spec → revisor-calidad → (+ verificador) → PR → puente**.
> Estado: ⬜ pendiente · 🟨 en curso · ✅ hecho · ⛔ bloqueado. Las fechas son de fusión en `main`.
> Entrega: **vie 30-oct, 23:59** (extendida desde el mar 27; meta interna por definir).

**Ruta crítica:** E0 (jue 22) → A5 datos reales (vie 23, 2 p. m.) → B4/D4 búsquedas (vie 23, noche) → B5 (sáb 24, mediodía)
→ C5 export real (sáb 24, noche) → E4 app final (dom 25) → E7 blog (lun 26).
**Nadie espera:** antes de cada entrega se trabaja con los datos sintéticos (núcleo §5.6) o con el JSON de juguete.

| Día | A — Datos (Daniel) | B — PD (Andrés) | C — Scorecard (Juan José) | D — LGD (Camila, José) | E — Producto (Daniel; José: app) |
|---|---|---|---|---|---|
| **Jue 22** | ✅ A1 descarga y conteos · ✅ A2 objetivo | Lee su spec; prepara entorno | Lee su spec; prepara entorno | Lee su spec; descarga el CSV | ✅ **E0** infraestructura (entorno, CI, run.py, blog, notebooks) + ruleset de `main` (sáb 24) |
| **Vie 23** | ✅ A3 variables (Daniel aprueba las 15) · ✅ A4 partición ≤2012/2013/2014 (Daniel) · ✅ **A5 datos reales (en `main` el sáb 24, 10:44 a. m., PR #1)** | ✅ B1 métricas (24-oct) · ✅ B2 MLP + entrenar (25-oct) · ✅ B3 logística (25-oct) (PR de `dev/andres`) · ✅ B4 Optuna (25-oct) | ✅ C1 scorecard (25-oct, `dev/juanjose`) · ✅ **C2 exportador + JSON real de la logística y casos de paridad** (25-oct, `dev/juanjose`) | ✅ D1 objetivo LGD (Camila) · ✅ D3 lineal + MLP (Camila) · ✅ D2 parquet LGD (Camila) (noche) | ✅ E1 esqueleto de app (José, PR #17) + Vercel (27-oct, PR #21–#22) · ✅ E10 `run.py notebooks` + plantilla · guion del video · revisión de PR |
| **Sáb 24** | ✅ A6 EDA inicial + hipótesis (vie) · ✅ A6b descriptivo después del preprocesamiento (vie) · coordina la grabación | ✅ **B5 30 semillas** (Andrés, lun 26) · ✅ B6 demo (Andrés, lun 26; 14–16 s; 5,0 s en el clon limpio, 28-oct) · ✅ B7 figuras (Andrés, lun 26; + aprendizaje, búsquedas y curva del video) | ✅ C3 scorecard logística (26-oct, `dev/juanjose`) · ✅ C4 SHAP del MLP + permutación + scorecard del MLP (26-oct, `dev/juanjose`) · ✅ **C5 export de los dos modelos** (26-oct, `dev/juanjose`) | ✅ D1b descriptivo de la LGD (Camila) · ✅ **D4 Optuna + 30 semillas** (Camila, dom 25) · ✅ **D4b rehecho** con L2 (λ/2)‖W‖², `evidencia` y semilla por val (José, lun 26, `dev/jose`) · ✅ D5 demo (Camila; 2,8–3,9 s en CPU, falta medir en otra máquina) · ✅ D6 figuras (Camila) | ✅ E2 app (formulario, PD, score, percentil; José, PR #17) · ⬜ **grabación del video con los 5** |
| **Dom 25** | ✅ A7 blog 01–03 (vie, borrador) · ✅ A8 bibliografía verificada (14; nuevas a pedido) · ✅ A9 notebook 01 (vie) | ✅ B8 blog 04–06 · ✅ B9 notebook 02 (lun 26) · ✅ B10 sensibilidad `grade`/`int_rate` | ✅ C6 hipótesis · ✅ C7 blog 07 · ✅ C8 notebook 03 | ✅ D7 blog 08 (Camila) · ✅ D8 notebook 04 (Camila) | ✅ E3 "¿y si…?" + enlaces (José, PR #17) · ✅ E4 paridad en CI (Daniel, 28-oct: la app entra al check `pruebas`) · ✅ E5 E[L] (PR #9; rehecha con la LGD de D4b, PR #15) · ✅ E6 video: Remotion con las 9 escenas y la curva de B (PR #16); publicado en YouTube el 30-oct (https://youtu.be/mdBuyLCzUUw) |
| **Lun 26** | Clon limpio · video individual | Video individual | Video individual | Video individual | 🟨 E7 blog publicado (00–13); faltan H-22…H-28 en el 12 y las marcas `revisar: autor` · ✅ E8 `run.py figures` integra todas las partes y el clon limpio pasa (Linux, 28-oct; `auditoria.yml`) · ✅ E9 desplegada (27-oct; sin prueba de usabilidad, Daniel 28-oct) · ✅ E11 notebooks 00 (demo pd) y 05 |
| **Mar 27–Jue 29** | Video individual · ✅ clon limpio (28-oct) | Video individual · ✅ B6 en otra máquina | Video individual | Video individual | ✅ Clips y edición del video · ✅ blog 12 · ✅ marcas del blog · ✅ URL del video en la app (30-oct) |
| **Vie 30** | — | — | — | — | **Entrega (23:59)** |

**Hasta el domingo 25 a las 8 p. m., todas las partes:** ≥ 1 hallazgo (o verificación documentada) en `alucinaciones/registro.md`.

## Cierre (todos)
| ID | Tarea | Fecha | Responsable | Estado |
|---|---|---|---|---|
| T5.1 | Cacería completa: ≥ 5 hallazgos, ≥ 3 categorías, ≥ 1 por parte | dom 25 | cazador-alucinaciones + todos | ✅ (21 confirmados; H-22…H-28 por confirmar) |
| T5.2 | Auditoría en clon limpio en otra máquina | antes del vie 30 | auditor-reproducibilidad + puente | ✅ (28-oct, runner Ubuntu de GitHub) |
| T5.3 | Videos individuales (2–3 min con evidencia concreta) | antes del vie 30 | cada integrante | ⬜ |
| T5.4 | Fichas de estudio + preguntas probables + predicciones en vivo | antes del vie 30 | tutor-sustentacion | 🟨 (predicciones de A–D en el notebook 00; faltan las fichas) |
| T5.5 | **Entrega** | **vie 30, 23:59** | todos | ⬜ |
| T5.6 | Simulacro de sustentación | antes de la sustentación | todos | ⬜ |
