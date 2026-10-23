# Fuentes de datos

| Dato | Fuente | URL | Fecha de descarga | SHA-256 | Tamaño | Quién | Notas |
|---|---|---|---|---|---|---|---|
| `loan/loan.csv` (créditos de Lending Club) | Kaggle, "Credit Risk Analysis" (ranadeep) | https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset | 2026-10-23 | `f01652b585e7f355d8749d0814b40068030241d201c6a31e6c9a3cb736eecccb` | 441 771 600 bytes | Daniel | 887 379 registros, 74 variables, `issue_d` de 2007-06 a 2015-12 (medido en `data/processed/resumen_datos.json`). **No se versiona** (excede el límite de GitHub). |
| `LCDataDictionary.xlsx` (diccionario de variables) | Mismo conjunto de Kaggle | ídem | 2026-10-23 | `6197339f12576ac7b56b281b4aca8dd47adb40f62b482794006f0073dd4a3654` | 20 995 bytes | Daniel | Hojas `LoanStats` (80 filas), `browseNotes`, `RejectStats`. Fuente de todas las definiciones citadas en `configs/features.yaml`. |

## Cómo reconstruir `data/processed/`
1. Descargar el conjunto desde la URL (requiere sesión en Kaggle) y descomprimirlo en `data/raw/credit-risk-dataset/`
   (deben quedar `data/raw/credit-risk-dataset/loan/loan.csv` y `data/raw/credit-risk-dataset/LCDataDictionary.xlsx`).
2. Comprobar el SHA-256 de la tabla de arriba.
3. `python scripts/preparar_datos.py` (no usa internet; tarda unos segundos).

## Observaciones del diccionario (verificadas el 23-oct-2026, útiles para la cacería)
- El diccionario describe `fico_range_low`, `fico_range_high`, `last_fico_range_*` e `is_inc_v`, que **no existen** en el CSV;
  en el CSV la verificación de ingresos se llama `verification_status`.
- El nombre `total_rev_hi_lim` aparece en el diccionario con un espacio de no separación al final.
- `verified_status_joint` (diccionario) se llama `verification_status_joint` en el CSV.
