# Blog del reporte — guía para redactar

El reporte es **una sola entrada de blog** (`docs/index.md`) que une las secciones de `secciones/`. Cada equipo escribe
**solo sus archivos** de `secciones/` (el comentario del inicio de cada archivo dice quién es responsable y qué exige el enunciado).

## Ver el blog mientras se escribe

```bash
.venv\Scripts\mkdocs serve -f blog/mkdocs.yml
```
Abre <http://127.0.0.1:8000> y se recarga al guardar. Antes de hacer commit:

```bash
.venv\Scripts\mkdocs build --strict -f blog/mkdocs.yml
```
Si falla, el mensaje dice qué regla se rompió. **No se fusiona un PR con el build en rojo.**

## Sintaxis

| Qué | Cómo se escribe |
|---|---|
| Ecuación en línea | `$f(\mathbf{x})$` |
| Ecuación en bloque | `$$ ... $$` en líneas aparte |
| Figura (APA) | ver abajo |
| Tabla (APA) | igual que la figura, con `::: tabla <id>` y una tabla Markdown dentro |
| Citar una figura o tabla | `[[fig:<id>]]` o `[[tab:<id>]]` → se convierte en "Figura N" / "Tabla N" con enlace |
| Cita entre paréntesis | `[@yurdakul2020]` → (Yurdakul & Naranjo, 2020) |
| Cita narrativa | `@yurdakul2020 definen…` → Yurdakul & Naranjo (2020) definen… |
| Varias citas / página | `[@a2020; @b2021]`, `[@yurdakul2020, p. 3]` |
| Nota para el equipo (no se publica) | `<!-- … -->` (las citas dentro de comentarios no cuentan) |

Figura (la numeración es automática; **título y fuente son obligatorios**):

```text
::: figura datos-anio
Título: Créditos sin desenlace y tasa de malos por año
Fuente: Elaboración propia a partir de data/processed/resumen_datos.json.
![Barras por año](figures/datos-anio.png)
:::
```
- Las imágenes van en `docs/figures/` y las genera el código (`python run.py figures`): **no se editan a mano**.
- Para una figura interactiva de Plotly: dentro del bloque, `incluir: figures/<archivo>.html`.
- El `id` usa minúsculas, números y guiones.

## Figuras ya generadas
`python run.py figures` las regenera en `docs/figures/`. Cada parte agrega aquí las suyas.

| id (para `[[fig:<id>]]`) | Archivo | Qué muestra | Parte |
|---|---|---|---|
| `datos-anio` | `datos-anio.png` | % sin desenlace y tasa de malos por año, con la partición | A |
| `eda-tasa-variables` | `eda-tasa-variables.png` | Tasa de malos por decil o categoría de cada variable (train) | A |
| `post-psi` | `post-psi.png` | PSI de cada variable frente a train | A |
| `post-correlacion` | `post-correlacion.png` | Correlaciones entre variables numéricas transformadas | A |
| `lgd-histograma` | `lgd-histograma.png` | Histograma de la LGD observada por conjunto y masa exacta en 0 y en 1 | D |
| `lgd-por-anio` | `lgd-por-anio.png` | % con LGD = 1 y LGD media por año de originación | D |
| `lgd-ead-plazo` | `lgd-ead-plazo.png` | EAD por plazo y por año, y LGD media por plazo | D |
| `lgd-distribucion-predicha` | `lgd-distribucion-predicha.png` | LGD predicha (lineal y MLP) frente a la observada en test | D |
| `lgd-qq` | `lgd-qq.png` | Q-Q de la LGD predicha frente a la observada en test | D |
| `lgd-error-por-banda` | `lgd-error-por-banda.png` | MAE por banda de LGD observada en test (lineal, MLP, constante mediana) | D |
| `lgd-semillas` | `lgd-semillas.png` | MAE de test de las 30 semillas del MLP frente a la lineal y las constantes | D |
| `el-bandas` | `el-bandas.png` | E[L] frente a pérdida realizada por banda de puntaje (solo créditos resueltos de test) | E |
| `el-pd-vs-tasa` | `el-pd-vs-tasa.png` | PD media predicha frente a tasa de malos observada por banda de puntaje (todo el test) | E |

## Bibliografía
Toda entrada de `references.bib` se copia del BibTeX oficial del DOI (`https://doi.org/<doi>` con
`Accept: application/x-bibtex`) y se contrasta con Crossref. Si necesitas una referencia nueva, pídela a quien lleva la
bibliografía (A8) o al `cazador-alucinaciones`; **nunca de memoria**.
