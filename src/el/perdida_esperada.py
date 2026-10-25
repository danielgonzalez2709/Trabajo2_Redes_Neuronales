"""Pérdida esperada por crédito (Parte E §3.2; enunciado, Parte 2, punto 5):  E[L]ᵢ = PDᵢ · LGDᵢ · EADᵢ.

- PD: predicción de la Parte B para cada crédito de test (`results/data/pd/pred_test.parquet`, columna pred_<modelo>).
- LGD: predicción de la Parte D para cada crédito de test, incluidos los que no incumplieron
  (`results/data/lgd/pred_pd_test.parquet`, núcleo §5.4).
- EAD de originación = `funded_amnt`, el monto desembolsado. Es conservadora: supone que el incumplimiento ocurre antes de
  amortizar nada. `resumen_el` reporta qué fracción del monto seguía pendiente, en promedio, en los que sí incumplieron.

Pérdida observada (para contrastar), en principal y neta de recuperaciones, sin intereses ni valor del dinero en el tiempo:
    pérdida = max(0, (funded_amnt − total_rec_prncp) − (recoveries − collection_recovery_fee))
Es la misma construcción de la LGD de la Parte D (D §3.2) multiplicada por la EAD al incumplimiento. En los créditos en
mora (`Late (31-120 days)`, `Default`) la cobranza no ha terminado: su pérdida observada es parcial y se marca.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ESTADOS_TERMINADOS = {"Fully Paid", "Charged Off", "Does not meet the credit policy. Status:Fully Paid",
                      "Does not meet the credit policy. Status:Charged Off"}


def perdida_esperada(pd_: np.ndarray, lgd: np.ndarray, ead: np.ndarray) -> np.ndarray:
    pd_, lgd, ead = (np.asarray(v, dtype=float) for v in (pd_, lgd, ead))
    if np.any((pd_ < 0) | (pd_ > 1)) or np.any((lgd < 0) | (lgd > 1)):
        raise ValueError("PD y LGD deben estar en [0, 1]")
    if np.any(ead < 0):
        raise ValueError("la EAD no puede ser negativa")
    return pd_ * lgd * ead


def perdida_observada(montos: pd.DataFrame) -> pd.DataFrame:
    pendiente = montos["funded_amnt"] - montos["total_rec_prncp"]
    recuperado = montos["recoveries"] - montos["collection_recovery_fee"]
    return pd.DataFrame({
        "perdida_observada": (pendiente - recuperado).clip(lower=0).to_numpy(),
        "observacion_completa": montos["loan_status"].isin(ESTADOS_TERMINADOS).to_numpy(),
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
    """Une por `id` y devuelve, por crédito de test: id, target, pd, lgd, ead, el, ead_incumplimiento (principal pendiente,
    `funded_amnt − total_rec_prncp`), perdida_observada y observacion_completa."""
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


def _totales(df: pd.DataFrame) -> dict:
    return {"n": int(len(df)), "el": float(df["el"].sum()), "ead": float(df["ead"].sum()),
            "perdida_observada": float(df["perdida_observada"].sum()),
            "tasa_malos": float(df["target"].mean()), "pd_media": float(df["pd"].mean()),
            "lgd_media": float(df["lgd"].mean())}


def resumen_el(el: pd.DataFrame, n_bandas: int = 10, banda: pd.Series | None = None) -> dict:
    """Totales y por banda. Sin `banda` (p. ej. bandas de puntaje de la Parte C), las bandas son deciles de PD."""
    grupos = banda if banda is not None else pd.qcut(el["pd"], q=n_bandas, labels=False, duplicates="drop")
    por_banda = [{"banda": int(b) if not isinstance(b, str) else b, **_totales(g)}
                 for b, g in el.groupby(grupos, observed=True)]
    por_banda.sort(key=lambda x: x["pd_media"])
    malos = el[el["target"] == 1]
    pendiente = malos["ead_incumplimiento"] / malos["ead"]  # qué tan conservadora es la EAD de originación
    completos = el[el["observacion_completa"]]
    return {
        "total": _totales(el),
        "total_observacion_completa": _totales(completos),
        "n_observacion_parcial": int((~el["observacion_completa"]).sum()),
        "por_banda": por_banda,
        "razon_el_sobre_perdida_observada": float(el["el"].sum() / el["perdida_observada"].sum())
        if el["perdida_observada"].sum() > 0 else None,
        "fraccion_pendiente_media_incumplidos": float(pendiente.mean()) if len(malos) else None,
    }
