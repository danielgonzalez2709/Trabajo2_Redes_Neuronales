# Parte C — Scorecard y variables de riesgo

| | |
|---|---|
| Responsables | **Juan José** (reparto de la pareja, 24-oct; Andrés hace la Parte B) |
| Rama personal | `dev/juanjose` |
| Carpetas propias | `src/scorecard/`, `src/export/`, `configs/scorecard/`, `src/viz/figs_scorecard.py`, `notebooks/03_scorecard.py`, `blog/secciones/07` |
| Lee del núcleo | §0, §3, §4.1, §5.2, §5.3, §5.4, §5.5 · **Anexo (enunciado):** Parte 1, tareas 2, 6 y 7; Entregables 1 y 4 |
| Arranca sin esperar | Sí: la scorecard es una fórmula sobre p y el exportador trabaja con un modelo de juguete en el formato del contrato |

## 1. Lo que pide el enunciado
- P1.6 **Scorecard**: definir puntaje base, odds de referencia y PDO; **derivar** la transformación de probabilidad a puntaje;
  explicar cómo se obtienen los puntos a partir del modelo (se permiten variables continuas).
- P1.7 **Variables de riesgo** con un método de interpretabilidad **justificado y citado**.
- P1.2 (cierre): decir qué hipótesis del análisis descriptivo se confirmaron y cuáles no.
- La app necesita puntaje, PD y percentil: el **exportador** (núcleo §5.5) es de esta parte.

## 2. Consumo / Entrego
| Consumo | De | Cuándo |
|---|---|---|
| Esqueleto y datos sintéticos | E | jue 22, noche |
| Hipótesis H1…Hn | A | sáb 24 |
| Modelo PD final + `pred_test` | B | sáb 24, mediodía (antes: modelo de juguete) |

| Entrego | A | Cuándo |
|---|---|---|
| `pd_model.json` **de juguete** en el formato del contrato | E (app) | **vie 23** |
| `scorecard.json`, SHAP, `hipotesis.json` | E (blog) | sáb 24 |
| `pd_model.json` **real** + `casos_paridad.json` | E (app) | **sáb 24, noche** |
| Sección 07 del blog | E | dom 25, 8 p. m. |

## 3. Especificación

### 3.1 Probabilidad → puntaje [PROPUESTO: valores por defecto]
Con odds de buen pagador o = (1 − p)/p (p = PD):
- score = offset + factor · ln(o), **factor = PDO / ln 2**, **offset = base − factor · ln(odds_base)**.
- Por defecto `base = 600` con `odds_base = 50` (50:1) y `pdo = 20`, todos configurables con `--set scorecard.*`.
- Mayor puntaje = menor riesgo. Como ln(o) = −logit(p): **score = offset − factor · logit(p)**.

### 3.2 Puntos por variable
- **Logística:** logit(p) = β₀ + Σ βⱼ xⱼ ⇒ puntosⱼ = −factor · βⱼ xⱼ y puntos base = offset − factor · β₀ (lineales en xⱼ
  estandarizada).
- **MLP:** SHAP φⱼ sobre el **logit**: logit(p) = φ₀ + Σ φⱼ ⇒ puntosⱼ = −factor · φⱼ y puntos base = offset − factor · φ₀. La
  suma de puntos reproduce el score exacto (aditividad; se comprueba en una prueba).
- [PENDIENTE: sáb 24, con B] ¿Qué modelo va a la app? Si el MLP no mejora de forma consistente, la logística es una opción
  legítima y se discute.

### 3.3 Interpretabilidad [PROPUESTO]
SHAP (Lundberg & Lee, 2017 [VERIFICAR]) global (media de |φ|) y por individuo + importancia por permutación en test como
contraste. Justificar por qué SHAP (aditividad, que conecta con los puntos de la scorecard).

### 3.4 Cierre de hipótesis
`hipotesis.json`: para cada H-k, evidencia del EDA (A) + dirección y magnitud en SHAP → **confirmada / refutada / no
concluyente**, con tabla en el blog.

### 3.5 Exportación (`run.py export`)
`pd_model.json` y `casos_paridad.json` exactamente según núcleo §5.5: W con forma salida × entrada, percentiles del score sobre
test, entradas de `features.yaml` con `en_app: true`. En las categóricas, `categorias` sin `OTRA` y `etiquetas_categorias`
copiado de `categorias_es` (núcleo §5.5, v0.12).

## 4. Tareas
| ID | Tarea | Depende | Aceptación | + Verificador | Fecha | Quién |
|---|---|---|---|---|---|---|
| C1 | `scorecard.py` (p → score) con `--set scorecard.*` | E0 | `test_scorecard.py`: score(odds_base) = base; duplicar odds suma PDO | **verificador-matematico** | vie 23 | Juan José |
| C2 | Exportador + `pd_model.json` de juguete | E0 | `test_export.py` (contrato §5.5); la app lo lee | revisor-spec | **vie 23** | Juan José |
| C3 | Scorecard de la logística | B3, C1 | Suma de puntos = score | verificador-matematico | sáb 24 | Juan José |
| C4 | SHAP del MLP + permutación + scorecard del MLP | B5, C1 | Aditividad comprobada numéricamente | verificador-matematico | sáb 24 | Juan José |
| C5 | Export real + casos de paridad | B5, C2 | Contrato §5.5 | revisor-spec | **sáb 24, noche** | Juan José |
| C6 | Cierre de hipótesis | A6, C4 | Cada H-k con veredicto y evidencia | analista-experimentos | dom 25 | Juan José |
| C8 | Notebook `03_scorecard` (núcleo §5.8) | C4, C6 | Corre en < 2 min; `SET = {"scorecard.pdo": 40}` cambia los puntajes | revisor-spec | dom 25 | Juan José |
| C7 | Blog 07 | C4, C6 | Derivación revisada; figuras citadas | redactor-tecnico + verificador-matematico | dom 25 | Juan José |

## 5. Pruebas de la parte
`test_scorecard.py` (base, PDO, monotonía, aditividad), `test_export.py`, `test_shap_aditividad.py`.

## 6. Puntos calientes de alucinación
- Odds malo:bueno en vez de bueno:malo; signo del factor; log10 en vez de ln; "PDO = puntos por cada 1 % de PD".
- SHAP aditivo en **probabilidad** (no lo es si se explica la salida sigmoide: se explica el logit).
- Explainer de SHAP para PyTorch: qué garantiza cada uno (DeepExplainer, GradientExplainer, KernelExplainer).
- Referencias de scorecards (p. ej. Siddiqi) con año o edición equivocados.

## 7. Blog y fundamentos
07: derivación completa de la scorecard (de logit a puntaje, factor y offset), puntos por variable para la logística y el MLP,
método de interpretabilidad citado, variables más riesgosas y tabla de hipótesis confirmadas/refutadas.

## 8. Definición de terminado
`python run.py scorecard --set scorecard.pdo=40` cambia los puntajes como se predice; `run.py export` produce el JSON real y la
prueba de paridad de la app pasa; ≥ 1 hallazgo o verificación documentada.
