"""Pérdida esperada por crédito (Parte E §3.2; enunciado, Parte 2, punto 5):  E[L]ᵢ = PDᵢ · LGDᵢ · EADᵢ.

- PD: predicción de la Parte B para cada crédito de test (`results/data/pd/pred_test.parquet`, columna pred_<modelo>).
- LGD: predicción de la Parte D para cada crédito de test, incluidos los que no incumplieron
  (`results/data/lgd/pred_pd_test.parquet`, núcleo §5.4).
- EAD de originación = `funded_amnt`, el monto desembolsado. Es conservadora: supone que el incumplimiento ocurre antes de
  amortizar nada. `resumen_el` reporta la media, en los castigados, de la fracción del monto que seguía pendiente.

Contraste con lo observado (E §3.2: "en créditos resueltos de test"), en principal y neto de recuperaciones, sin
intereses ni valor del dinero en el tiempo. Según `loan_status`:
- Castigados (`Charged Off` y `Does not meet the credit policy. Status:Charged Off`; la población de la LGD de D §3.2):
  pérdida realizada = max(0, (funded_amnt − total_rec_prncp) − (recoveries − collection_recovery_fee)).
- Pagados (`Fully Paid` y `Does not meet the credit policy. Status:Fully Paid`): pérdida realizada 0. Si
  total_rec_prncp < funded_amnt, la diferencia es principal no registrado: se cuenta como anomalía de los datos
  (`principal_no_registrado`) y NO se suma a la pérdida.
- En mora (`Late (31-120 days)`, `Default`): la cobranza no terminó, así que no hay pérdida realizada (NaN). Se guarda
  aparte una pérdida PROVISIONAL con la misma cuenta, max(0, pendiente − (recoveries − fee)); como en estos datos no
  tienen recuperaciones, equivale a suponer LGD = 1 sobre el principal pendiente. No es una cota inferior de la pérdida
  final ni una LGD observada.
Los resueltos (castigados y pagados) tienen `observacion_completa` = True; solo con ellos se calcula la razón
E[L] / pérdida realizada. Cualquier otro estado es un error (no llega a `loans_pd.parquet`: su objetivo es NA, A §3.3).

Bandas: cuantiles del puntaje de la scorecard de la Parte C (deciles con `bandas.n = 10`; `bandas_de_puntaje`),
numerados 1…n de menor puntaje (más riesgo) a mayor; `resumen_el` las lee de la columna `banda`, como lee el puntaje
de `score`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Estados de `loan_status` con objetivo definido (A §3.3), agrupados para el contraste
CASTIGADOS = {"Charged Off", "Does not meet the credit policy. Status:Charged Off"}  # población de la LGD (D §3.2)
PAGADOS = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}
EN_MORA = {"Late (31-120 days)", "Default"}  # cobranza abierta (D §3.2 los excluye)
ESTADO = {**{e: "castigado" for e in CASTIGADOS}, **{e: "pagado" for e in PAGADOS}, **{e: "en_mora" for e in EN_MORA}}


def perdida_esperada(pd_: np.ndarray, lgd: np.ndarray, ead: np.ndarray) -> np.ndarray:
    """E[L]ᵢ = PDᵢ · LGDᵢ · EADᵢ, con PD y LGD en [0, 1] y EAD ≥ 0, todos finitos; si no, ValueError."""
    pd_, lgd, ead = (np.asarray(v, dtype=float) for v in (pd_, lgd, ead))
    if not all(np.all(np.isfinite(v)) for v in (pd_, lgd, ead)):  # un NaN pasaría las comparaciones de rango
        raise ValueError("PD, LGD y EAD deben ser finitas (sin NaN ni ±inf)")
    if np.any((pd_ < 0) | (pd_ > 1)) or np.any((lgd < 0) | (lgd > 1)):
        raise ValueError("PD y LGD deben estar en [0, 1]")
    if np.any(ead < 0):
        raise ValueError("la EAD no puede ser negativa")
    return pd_ * lgd * ead


def perdida_observada(montos: pd.DataFrame) -> pd.DataFrame:
    """Por crédito: estado (castigado, pagado, en_mora), perdida_observada (realizada; NaN en mora),
    perdida_provisional (solo en mora), principal_no_registrado (solo pagados) y observacion_completa."""
    desconocidos = sorted(set(montos["loan_status"]) - set(ESTADO))
    if desconocidos:
        raise ValueError(f"loan_status sin tratamiento en la E[L]: {desconocidos}")
    estado = montos["loan_status"].map(ESTADO)
    pendiente = montos["funded_amnt"] - montos["total_rec_prncp"]
    neta = (pendiente - (montos["recoveries"] - montos["collection_recovery_fee"])).clip(lower=0)
    castigado, pagado, mora = (estado == "castigado"), (estado == "pagado"), (estado == "en_mora")
    return pd.DataFrame({
        "estado": estado.to_numpy(),
        "perdida_observada": np.select([castigado, pagado], [neta, 0.0], np.nan),  # en mora: no realizada
        "perdida_provisional": np.where(mora, neta, np.nan),  # LGD = 1 sobre el pendiente (sin recuperaciones)
        "principal_no_registrado": np.where(pagado, pendiente.clip(lower=0), np.nan),  # anomalía de los datos
        "observacion_completa": (castigado | pagado).to_numpy(),
    }, index=montos.index)


def _columna(pred: pd.DataFrame, ids: pd.Series, modelo: str, nombre: str) -> np.ndarray:
    columna = f"pred_{modelo}"
    if columna not in pred.columns:
        raise ValueError(f"{nombre}: falta la columna {columna} (contrato del núcleo §5.4)")
    valores = pred.set_index("id")[columna].reindex(ids)
    if valores.isna().any():
        raise ValueError(f"{nombre}: {int(valores.isna().sum())} créditos de test sin predicción")
    return valores.to_numpy()


def calcular_el(test: pd.DataFrame, pred_pd: pd.DataFrame, pred_lgd: pd.DataFrame, montos: pd.DataFrame,
                modelo_pd: str, modelo_lgd: str) -> pd.DataFrame:
    """Une por `id` y devuelve, por crédito de test: id, target, pd, lgd, ead, el, ead_incumplimiento (principal
    pendiente, `funded_amnt − total_rec_prncp`) y las columnas de `perdida_observada`."""
    ids = test["id"].astype(str).reset_index(drop=True)
    m = montos.assign(id=montos["id"].astype(str)).set_index("id").reindex(ids)
    if m["funded_amnt"].isna().any():
        raise ValueError(f"montos: {int(m['funded_amnt'].isna().sum())} créditos de test sin monto")
    pd_ = _columna(pred_pd.assign(id=pred_pd["id"].astype(str)), ids, modelo_pd, "PD")
    lgd = _columna(pred_lgd.assign(id=pred_lgd["id"].astype(str)), ids, modelo_lgd, "LGD")
    ead = m["funded_amnt"].to_numpy(dtype=float)
    obs = perdida_observada(m.reset_index(drop=True))
    return pd.DataFrame({"id": ids, "target": test["target"].to_numpy(), "pd": pd_, "lgd": lgd, "ead": ead,
                         "el": perdida_esperada(pd_, lgd, ead),
                         "ead_incumplimiento": (m["funded_amnt"] - m["total_rec_prncp"]).to_numpy(dtype=float),
                         **{c: obs[c].to_numpy() for c in obs.columns}})


def bandas_de_puntaje(score: pd.Series, n_bandas: int) -> pd.Series:
    """Cuantiles del puntaje numerados 1…k (k = n_bandas, o menos si los empates fusionan bordes), con el índice de
    `score`: banda 1 = menor puntaje = más riesgo. Con todos los puntajes iguales no hay bandas: ValueError."""
    if score.nunique() < 2:
        raise ValueError("todos los puntajes son iguales: no se pueden formar bandas de puntaje")
    return pd.qcut(score, q=n_bandas, labels=False, duplicates="drop").astype(int) + 1


def _razon(el: float, observada: float) -> float | None:
    # E[L] / pérdida realizada; sin pérdida realizada la razón no está definida
    return el / observada if observada > 0 else None


def _media(serie: pd.Series) -> float | None:
    # None en vez de NaN si no hay créditos (el JSON no admite NaN)
    return float(serie.mean()) if len(serie) else None


def _base(df: pd.DataFrame) -> dict:
    return {"n": int(len(df)), "el": float(df["el"].sum()), "ead": float(df["ead"].sum()),
            "tasa_malos": _media(df["target"]), "pd_media": _media(df["pd"]), "lgd_media": _media(df["lgd"])}


def _bloques(df: pd.DataFrame) -> dict:
    """total (todos), resueltos (contraste con la pérdida realizada) y en_mora (pérdida provisional aparte)."""
    resueltos = df[df["observacion_completa"]]
    mora = df[df["estado"] == "en_mora"]
    realizada = float(resueltos["perdida_observada"].sum())
    return {
        "total": _base(df),
        "resueltos": {**_base(resueltos), "perdida_observada": realizada,
                      "razon_el_sobre_perdida_observada": _razon(float(resueltos["el"].sum()), realizada)},
        "en_mora": {**_base(mora), "perdida_provisional": float(mora["perdida_provisional"].sum())},
    }


def resumen_el(el: pd.DataFrame) -> dict:
    """Totales y por banda (columna `banda` de `el`, p. ej. `bandas_de_puntaje(el["score"], n)`; groupby las ordena
    por número), cada uno con los bloques total, resueltos y en_mora; cada banda reporta además su rango de puntaje.
    Incluye la anomalía de los pagados con principal no registrado y la fracción pendiente media de los castigados."""
    por_banda = [{"banda": int(b), "score_min": float(g["score"].min()), "score_max": float(g["score"].max()),
                  **_bloques(g)} for b, g in el.groupby("banda")]
    castigados = el[el["estado"] == "castigado"]
    pagados = el[el["estado"] == "pagado"]
    anomalos = pagados[pagados["principal_no_registrado"] > 0]
    return {
        **_bloques(el),
        "anomalia_pagados": {
            "descripcion": f"pagados ({' y '.join(sorted(PAGADOS))}) con total_rec_prncp < funded_amnt; "
                           "no se suma a la pérdida",
            "n": int(len(anomalos)), "principal_no_registrado": float(anomalos["principal_no_registrado"].sum())},
        "por_banda": por_banda,
        # media por crédito castigado de (funded_amnt − total_rec_prncp) / funded_amnt: qué tan conservadora es la EAD
        "fraccion_principal_pendiente_media_castigados": _media(castigados["ead_incumplimiento"] / castigados["ead"]),
    }
