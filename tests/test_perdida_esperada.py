"""E5 — Pérdida esperada E[L] = PD · LGD · EAD (Parte E §3.2; enunciado, Parte 2, punto 5)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.el.perdida_esperada import calcular_el, perdida_esperada, perdida_observada, resumen_el


def test_formula_a_mano():
    el = perdida_esperada(np.array([0.1, 0.5, 0.0]), np.array([0.8, 0.4, 0.9]), np.array([10_000, 2_000, 5_000]))
    assert el.tolist() == pytest.approx([800.0, 400.0, 0.0])


@pytest.mark.parametrize("pd_, lgd, ead", [([1.2], [0.5], [100]), ([0.2], [-0.1], [100]), ([0.2], [0.5], [-1])])
def test_valores_fuera_de_rango_son_error(pd_, lgd, ead):
    with pytest.raises(ValueError):
        perdida_esperada(np.array(pd_), np.array(lgd), np.array(ead))


def test_perdida_observada_a_mano():
    montos = pd.DataFrame({
        "loan_status": ["Fully Paid", "Charged Off", "Charged Off", "Late (31-120 days)"],
        "funded_amnt": [10_000.0, 10_000.0, 5_000.0, 8_000.0],
        "total_rec_prncp": [10_000.0, 4_000.0, 1_000.0, 2_000.0],
        "recoveries": [0.0, 1_500.0, 6_000.0, 0.0],
        "collection_recovery_fee": [0.0, 300.0, 100.0, 0.0],
    })
    obs = perdida_observada(montos)
    # pendiente − (recuperaciones − tarifa), sin negativos: 0; 6000 − 1200 = 4800; 4000 − 5900 → 0; 6000 − 0 = 6000
    assert obs["perdida_observada"].tolist() == pytest.approx([0.0, 4_800.0, 0.0, 6_000.0])
    # la cobranza de un crédito en mora todavía no terminó: su pérdida observada es parcial
    assert obs["observacion_completa"].tolist() == [True, True, True, False]


def _entradas(n=2000, semilla=0):
    rng = np.random.default_rng(semilla)
    ids = [str(i) for i in range(n)]
    test = pd.DataFrame({"id": ids, "target": (rng.random(n) < 0.25).astype("int8")})
    pd_pred = pd.DataFrame({"id": ids, "pred_mlp": rng.uniform(0.02, 0.6, n), "pred_logistica": rng.uniform(0, 1, n)})
    lgd_pred = pd.DataFrame({"id": ids[::-1], "pred_mlp": rng.uniform(0.5, 1.0, n)})  # otro orden: se une por id
    montos = pd.DataFrame({"id": ids, "loan_status": np.where(test["target"] == 1, "Charged Off", "Fully Paid"),
                           "funded_amnt": rng.uniform(1_000, 35_000, n)})
    montos["total_rec_prncp"] = np.where(test["target"] == 1, montos["funded_amnt"] * 0.3, montos["funded_amnt"])
    montos["recoveries"] = np.where(test["target"] == 1, 500.0, 0.0)
    montos["collection_recovery_fee"] = np.where(test["target"] == 1, 50.0, 0.0)
    return test, pd_pred, lgd_pred, montos


def test_calcular_el_une_por_id_y_usa_ead_de_originacion():
    test, pd_pred, lgd_pred, montos = _entradas()
    el = calcular_el(test, pd_pred, lgd_pred, montos, modelo_pd="mlp", modelo_lgd="mlp")
    assert len(el) == len(test) and el["id"].tolist() == test["id"].tolist()
    fila = el.iloc[7]
    lgd_7 = lgd_pred.set_index("id").loc["7", "pred_mlp"]
    assert fila["lgd"] == pytest.approx(lgd_7)
    assert fila["ead"] == pytest.approx(montos.loc[7, "funded_amnt"])
    assert fila["el"] == pytest.approx(fila["pd"] * fila["lgd"] * fila["ead"])


def test_calcular_el_falla_si_falta_un_credito():
    test, pd_pred, lgd_pred, montos = _entradas()
    with pytest.raises(ValueError, match="sin predicción"):
        calcular_el(test, pd_pred.iloc[:-1], lgd_pred, montos, modelo_pd="mlp", modelo_lgd="mlp")


def test_resumen_suma_por_bandas_igual_al_total():
    test, pd_pred, lgd_pred, montos = _entradas()
    el = calcular_el(test, pd_pred, lgd_pred, montos, modelo_pd="mlp", modelo_lgd="mlp")
    r = resumen_el(el, n_bandas=10)
    assert r["total"]["el"] == pytest.approx(el["el"].sum())
    assert sum(b["el"] for b in r["por_banda"]) == pytest.approx(r["total"]["el"])
    assert sum(b["n"] for b in r["por_banda"]) == len(el)
    # bandas ordenadas por PD creciente
    pds = [b["pd_media"] for b in r["por_banda"]]
    assert pds == sorted(pds)
    # en _entradas los malos amortizaron el 30 % del principal: queda pendiente el 70 %
    assert r["fraccion_pendiente_media_incumplidos"] == pytest.approx(0.7)


def test_comando_el_avisa_si_faltan_las_entregas(tmp_path, capsys, monkeypatch):
    """Sin las predicciones de B y D el comando termina con 2, dice qué falta y no escribe nada."""
    from pathlib import Path

    from src.common.config import load_config
    from src.el.experimento import manejador

    raiz = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(raiz)
    cfg = load_config(raiz / "configs" / "el" / "base.yaml")
    cfg["pd"]["predicciones"] = str(tmp_path / "no_existe_pd.parquet")
    cfg["lgd"]["predicciones"] = str(tmp_path / "no_existe_lgd.parquet")
    cfg["salida"] = str(tmp_path / "salida")
    assert manejador(cfg) == 2
    salida = capsys.readouterr().out
    assert "Parte B" in salida and "Parte D" in salida
    assert not (tmp_path / "salida").exists()


def test_correr_el_de_punta_a_punta(tmp_path):
    from src.el.experimento import correr_el

    test, pd_pred, lgd_pred, montos = _entradas()
    test = test.assign(split="test")
    rutas = {n: tmp_path / f"{n}.parquet" for n in ("loans", "montos", "pd", "lgd")}
    test.to_parquet(rutas["loans"]); montos.to_parquet(rutas["montos"])
    pd_pred.to_parquet(rutas["pd"]); lgd_pred.to_parquet(rutas["lgd"])
    cfg = {"datos": {"loans": rutas["loans"], "montos": rutas["montos"]},
           "pd": {"predicciones": rutas["pd"], "modelo": "mlp"}, "lgd": {"predicciones": rutas["lgd"], "modelo": "mlp"},
           "bandas": {"n": 10}, "salida": tmp_path / "el"}
    r = correr_el(cfg, Path("/"))
    assert (tmp_path / "el" / "summary.json").is_file() and (tmp_path / "el" / "el_test.parquet").is_file()
    assert r["total"]["n"] == len(test)
