# Parte E — Integración y producto (puente)

| | |
|---|---|
| Responsables | **Daniel y Camila (Camila: frontend de la app)** |
| Ramas personales | `dev/daniel`, `dev/camila` |
| Carpetas propias | raíz del repo, `.github/`, `src/common/` (menos `preprocesamiento.py` y `metricas.py`), `src/datos/sintetico.py`, `src/el/`, `src/viz/theme.py`, `src/viz/style.py`, `src/viz/figs_el.py`, `app/`, `video/`, `blog/` (estructura y secciones 00, 09–13), `notebooks/` (infraestructura, `00_sustentacion`, `05_perdida_esperada`) |
| Lee del núcleo | Todo, incluido el **Anexo (enunciado)** completo |

## 1. Lo que pide el enunciado
- P2.5 **Pérdida esperada** E[L] = PD · LGD · EAD por crédito, **definiendo y justificando la EAD**.
- **App publicada**: la persona ingresa sus características y obtiene su puntaje y su PD, su ubicación frente a la población
  (p. ej. percentil) y cómo cambiaría su puntaje si cambiara alguna característica; intuitiva; enlaces al reporte y al video;
  se levanta localmente con un comando.
- **Reporte-blog** con APA 7, figuras y tablas numeradas y citadas, uso de IA, cacería y enlaces a repo, app y video.
- **Video promocional** de 2–3 min con los 5 integrantes.
- **Repositorio reproducible**, todo corre en vivo, prueba en un clon limpio en otra máquina.

## 2. Consumo / Entrego
| Entrego | A | Cuándo |
|---|---|---|
| Repo en `main` con ruleset, CI, código común del Trabajo 1, preprocesador mínimo, datos sintéticos (núcleo §5.6), blog vacío, esqueleto de la app | todos | **jue 22, noche** |
| Revisión de PR (mañana, tarde y noche) | todos | continuo |

| Consumo | De | Cuándo |
|---|---|---|
| `pd_model.json` de juguete / real | C | vie 23 / sáb 24, noche |
| `pred_test` de PD y LGD | B, D | sáb 24 |
| Secciones del blog | todos | dom 25, 8 p. m. |
| Clips del video | todos | sáb 24, tarde |

## 3. Especificación

### 3.1 Infraestructura (E0, hoy)
- Esqueleto de núcleo §4, `requirements.txt` fijado (Python 3.12), README con instalación (CPU y nota para GPU), CODEOWNERS,
  plantilla de PR, CI (`pytest -m "not lento"` + prueba de paridad cuando exista).
- Portar del Trabajo 1 `config.py`, `cli.py`, `seeds.py` (añadiendo torch), `results.py`, `test_no_network.py` y el blog
  (mkdocs.yml, `hooks/reporte.py`, `apa.csl`, pruebas). `run.py` con los subcomandos de núcleo §4.1 (vacíos al inicio).
- `src/datos/sintetico.py` (núcleo §5.6) y la versión mínima del `Preprocesador` (núcleo §5.2) con su firma definitiva.
- Ruleset de `main` (manual en GitHub, después del primer push).
- `requirements.txt` incluye `jupytext`, `nbclient`, `ipykernel` y `jupyterlab` para los notebooks (núcleo §5.8).

### 3.2 Pérdida esperada [PROPUESTO]
- E[L]ᵢ = PDᵢ · LGDᵢ · EADᵢ en los créditos de test, con PD de B y LGD de D.
- EAD de originación: **monto desembolsado (`funded_amnt`)**. Es conservadora (supone incumplimiento antes de amortizar). Se
  discute como alternativa la exposición promedio según el plazo.
- Salidas: E[L] total y por banda de puntaje; contraste con la pérdida observada en créditos resueltos de test.

### 3.3 App
| ID | Requisito | Aceptación |
|---|---|---|
| R-APP-1 | Formulario | Entradas del contrato, etiquetas y ayudas en español, rangos válidos |
| R-APP-2 | Puntaje y PD | Al instante; coinciden con Python (prueba de paridad) |
| R-APP-3 | Ubicación en la población | Percentil + histograma de la población con su marca |
| R-APP-4 | "¿Y si…?" | Deslizadores por variable con el puntaje nuevo en vivo + los 3 cambios que más suben el puntaje |
| R-APP-5 | Intuitiva | Probada con 2 personas ajenas al curso antes del lun 26 |
| R-APP-6 | Enlaces | Al blog y al video, visibles |
| R-APP-7 | Local con un comando | `npm --prefix app run dev` |
| R-APP-8 | Publicada | URL de Vercel en el blog y el README |

Arquitectura: Next.js + TypeScript con `output: "export"` (sitio estático); `app/lib/preprocesar.ts` y `app/lib/modelo.ts`
(forward de ~15 líneas) leen el JSON del contrato. Vercel Hobby con *Root Directory* = `app/`; sin despliegue cuando un commit
no toca `app/` [VERIFICAR la opción en el panel]; vista previa por PR. Aviso visible: herramienta académica, no es una oferta
de crédito. Diseño con la skill `apple-design`.

### 3.4 Blog
Material for MkDocs 1.6.x + hook del Trabajo 1 (el build falla con figuras sin título o fuente, figuras no citadas o citas
inexistentes). Secciones:

| # | Sección | Responsable |
|---|---|---|
| 00 | Resumen y enlaces (repo, app, video) | E |
| 01–03 | Introducción y caso de uso · Datos · Análisis descriptivo e hipótesis | A |
| 04–06 | Fundamentos de redes · Métricas · Resultados de PD y comparación justa | B |
| 07 | Scorecard y variables de riesgo | C |
| 08 | LGD | D |
| 09 | Pérdida esperada | E |
| 10 | Discusión y aprendizajes | todos (consolida E) |
| 11 | Uso de IA | E + todos |
| 12 | Cacería de la alucinación | E + todos |
| 13 | Reproducibilidad y referencias | E |

### 3.5 Video promocional
Remotion en `video/` con clips de **los 5** (grabación conjunta el sáb 24, coordinada por A). Estética de keynote: tipografía
grande, fondo oscuro, la app dentro de un dispositivo, cifras animadas. Guion: problema → solución → demo → cómo funciona →
equipo → enlace. 2–3 min. YouTube no listado; enlace en la app y el blog.

## 4. Tareas
| ID | Tarea | Depende | Aceptación | + Verificador | Fecha | Quién |
|---|---|---|---|---|---|---|
| E0 | Infraestructura (§3.1) + primer push + ruleset | — | `pip install` limpio; pruebas portadas pasan; un push directo a `main` de otra cuenta es rechazado | auditor-reproducibilidad | **jue 22** | Daniel |
| E1 | Esqueleto de la app + Vercel enlazado | E0 | `npm --prefix app run dev` levanta; URL de vista previa | implementador-web | vie 23 | Camila |
| E2 | App: formulario, PD, score, percentil (con el JSON de juguete) | C2 | R-APP-1 a R-APP-3 | implementador-web | sáb 24 | Camila |
| E3 | App: "¿y si…?" + enlaces | E2 | R-APP-4, R-APP-6 | implementador-web | dom 25 | Camila |
| E4 | Paridad en CI + modelo real | C5, E3 | 1 000/1 000 dentro de la tolerancia | revisor-spec | dom 25 | Camila |
| E5 | E[L] | B5, D4 | Por crédito y por banda; contraste con lo observado | verificador-matematico | dom 25 | Daniel |
| E6 | Video: guion (vie 23), grabación (sáb 24), edición en Remotion (dom 25) | E2 | 2–3 min, aparecen los 5 | — | dom 25 | Daniel |
| E7 | Blog 00, 09–13 + publicación en GitHub Pages | todas | `mkdocs build --strict`; enlaces a repo, app y video | redactor-tecnico + cazador | lun 26 | Daniel |
| E8 | `run.py figures` integra todas las figuras | B7, C7, D6 | Clon limpio regenera todo | auditor-reproducibilidad | lun 26 | Daniel |
| E10 | `run.py notebooks` (.py → .ipynb + ejecución) + plantilla con la celda de parámetros | E0 | La plantilla corre con los sintéticos; `test_notebooks.py` | revisor-spec | vie 23 | Daniel |
| E11 | Notebooks `00_sustentacion` y `05_perdida_esperada` | E5, B9, C8, D8 | Cada uno en < 2 min; `00` trae la celda de predicción escrita antes de ejecutar | revisor-spec | lun 26 | Daniel |
| E9 | Usabilidad con 2 personas + despliegue final | E4 | R-APP-5, R-APP-8 | — | lun 26 | Camila |

## 5. Puntos calientes de alucinación
- Límites y precios de Vercel, licencia de Remotion, mantenimiento de MkDocs Material (V-01…V-10 del registro).
- Transposición de W y orden de columnas al portar el forward a TypeScript.
- E[L] con EAD de castigo en lugar de EAD de originación (o al revés) sin justificarlo.

## 6. Definición de terminado
App publicada y probada; blog publicado con todos los enlaces; video subido; clon limpio en otra máquina en verde; cacería con
≥ 5 hallazgos en ≥ 3 categorías.
