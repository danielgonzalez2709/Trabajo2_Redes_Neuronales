"""E5 — Pérdida esperada E[L] = PD · LGD · EAD (Parte E §3.2; enunciado, Parte 2, punto 5).

Contraste con lo observado SOLO en los créditos resueltos de test (E §3.2): castigados (pérdida realizada) y pagados
(pérdida 0). Los créditos en mora llevan una pérdida provisional aparte; los pagados con principal sin registrar se
cuentan como anomalía de los datos, sin sumarlos a la pérdida.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.el.perdida_esperada import (CASTIGADOS, EN_MORA, PAGADOS, bandas_de_puntaje, calcular_el, perdida_esperada,
                                     perdida_observada, resumen_el)

RAIZ = Path(__file__).resolve().parents[1]
CO_DNM = "Does not meet the credit policy. Status:Charged Off"
FP_DNM = "Does not meet the credit policy. Status:Fully Paid"


def test_formula_a_mano():
    el = perdida_esperada(np.array([0.1, 0.5, 0.0]), np.array([0.8, 0.4, 0.9]), np.array([10_000, 2_000, 5_000]))
    assert el.tolist() == pytest.approx([800.0, 400.0, 0.0])


@pytest.mark.parametrize("pd_, lgd, ead", [([1.2], [0.5], [100]), ([-0.1], [0.5], [100]), ([0.2], [1.1], [100]),
                                           ([0.2], [-0.1], [100]), ([0.2], [0.5], [-1])])
def test_valores_fuera_de_rango_son_error(pd_, lgd, ead):
    with pytest.raises(ValueError):
        perdida_esperada(np.array(pd_), np.array(lgd), np.array(ead))


@pytest.mark.parametrize("pd_, lgd, ead", [([np.nan], [0.5], [100]), ([0.2], [np.nan], [100]), ([0.2], [0.5], [np.nan]),
                                           ([0.2], [0.5], [np.inf]), ([np.inf], [0.5], [100]),
                                           ([0.2], [-np.inf], [100])])
def test_valores_no_finitos_son_error(pd_, lgd, ead):
    """Un NaN pasaría las comparaciones de rango y daría una E[L] NaN en silencio."""
    with pytest.raises(ValueError, match="finit"):
        perdida_esperada(np.array(pd_), np.array(lgd), np.array(ead))


def test_bordes_aceptados():
    """PD y LGD en [0, 1] cerrado y EAD ≥ 0: los bordes son válidos (mata los mutantes > → >= y < → <=)."""
    el = perdida_esperada(np.array([0.0, 1.0, 1.0, 1.0, 0.5]), np.array([1.0, 0.0, 1.0, 1.0, 0.5]),
                          np.array([100.0, 100.0, 100.0, 0.0, 100.0]))
    assert el.tolist() == pytest.approx([0.0, 0.0, 100.0, 0.0, 25.0])


# ---------------------------------------------------------------- pérdida realizada, provisional y anomalía

def _montos_a_mano():
    return pd.DataFrame({
        "loan_status": ["Fully Paid", "Fully Paid", FP_DNM, "Charged Off", "Charged Off", CO_DNM,
                        "Late (31-120 days)", "Default"],
        "funded_amnt": [10_000.0, 10_000.0, 4_000.0, 10_000.0, 5_000.0, 3_000.0, 8_000.0, 6_000.0],
        "total_rec_prncp": [10_000.0, 9_950.0, 4_000.0, 4_000.0, 1_000.0, 1_000.0, 2_000.0, 1_000.0],
        "recoveries": [0.0, 0.0, 0.0, 1_500.0, 6_000.0, 200.0, 0.0, 0.0],
        "collection_recovery_fee": [0.0, 0.0, 0.0, 300.0, 100.0, 0.0, 0.0, 0.0],
    })


def test_perdida_observada_a_mano():
    obs = perdida_observada(_montos_a_mano())
    assert obs["estado"].tolist() == ["pagado", "pagado", "pagado", "castigado", "castigado", "castigado", "en_mora",
                                      "en_mora"]
    # pérdida realizada solo en los castigados: max(0, pendiente − (recuperaciones − tarifa));
    # 6000 − 1200 = 4800; 4000 − 5900 → 0; 2000 − 200 = 1800. Los pagados: 0 (también el de 50 USD sin registrar).
    realizada = obs["perdida_observada"].tolist()
    assert realizada[:6] == pytest.approx([0.0, 0.0, 0.0, 4_800.0, 0.0, 1_800.0])
    # en mora la cobranza no terminó: no hay pérdida realizada; va aparte como provisional (LGD = 1 sobre el pendiente)
    assert all(math.isnan(v) for v in realizada[6:])
    provisional = obs["perdida_provisional"].tolist()
    assert all(math.isnan(v) for v in provisional[:6]) and provisional[6:] == pytest.approx([6_000.0, 5_000.0])
    # anomalía: pagado con principal sin registrar (funded − total_rec_prncp > 0); solo se define en los pagados
    anomalia = obs["principal_no_registrado"].tolist()
    assert anomalia[:3] == pytest.approx([0.0, 50.0, 0.0]) and all(math.isnan(v) for v in anomalia[3:])
    assert obs["observacion_completa"].tolist() == [True] * 6 + [False] * 2


def test_estado_desconocido_es_error():
    montos = _montos_a_mano()
    montos.loc[0, "loan_status"] = "Current"
    with pytest.raises(ValueError, match="Current"):
        perdida_observada(montos)


def test_estados_coinciden_con_los_de_la_lgd_de_D_y_el_objetivo_de_A():
    """Castigados = población de la LGD (D §3.2); en mora = los que D excluye; todos son estados con objetivo (A)."""
    from src.datos.objetivo import MAPEO_LOAN_STATUS
    from src.lgd.objetivo import ESTADOS_EXCLUIDOS, ESTADOS_LGD

    assert CASTIGADOS == set(ESTADOS_LGD) and EN_MORA == set(ESTADOS_EXCLUIDOS)
    con_objetivo = {e for e, y in MAPEO_LOAN_STATUS.items() if y is not None}  # los demás no llegan a loans_pd
    assert CASTIGADOS | EN_MORA | PAGADOS == con_objetivo
    assert all(MAPEO_LOAN_STATUS[e] == 0 for e in PAGADOS)
    assert all(MAPEO_LOAN_STATUS[e] == 1 for e in CASTIGADOS | EN_MORA)


# ---------------------------------------------------------------- unión y resumen

def _entradas(n=2000, semilla=0):
    """Test sintético: malos castigados (amortizaron 30 %) y en mora (cada 5.º malo; amortizaron 50 %); 1 de cada 10
    pagados con 10 USD de principal sin registrar. Como en la vida real, las tablas no vienen alineadas: `montos` está
    barajado y `montos` y la PD traen los ids como enteros (el test, como texto): `calcular_el` debe unir por id."""
    rng = np.random.default_rng(semilla)
    ids = [str(i) for i in range(n)]
    target = (rng.random(n) < 0.25).astype("int8")
    test = pd.DataFrame({"id": ids, "target": target})
    pd_pred = pd.DataFrame({"id": np.arange(n), "pred_mlp": rng.uniform(0.02, 0.6, n),
                            "pred_logistica": rng.uniform(0, 1, n)})  # ids enteros
    lgd_pred = pd.DataFrame({"id": ids[::-1], "pred_mlp": rng.uniform(0.5, 1.0, n)})  # otro orden: se une por id
    malo = target == 1
    mora = malo & (np.cumsum(malo) % 5 == 0)
    castigado = malo & ~mora
    anomalo = ~malo & (np.arange(n) % 10 == 0)
    funded = rng.uniform(1_000, 35_000, n)
    estado = np.select([castigado, mora], ["Charged Off", "Late (31-120 days)"], "Fully Paid")
    montos = pd.DataFrame({"id": np.arange(n), "loan_status": estado, "funded_amnt": funded})  # ids enteros
    montos["total_rec_prncp"] = np.select([castigado, mora, anomalo], [funded * 0.3, funded * 0.5, funded - 10.0],
                                          funded)
    montos["recoveries"] = np.where(castigado, 500.0, 0.0)
    montos["collection_recovery_fee"] = np.where(castigado, 50.0, 0.0)
    montos = montos.iloc[rng.permutation(n)].reset_index(drop=True)  # otro orden: se une por id
    return test, pd_pred, lgd_pred, montos


def _alinear(montos, el):
    """Filas de `montos` en el orden de los créditos de `el` (para comparar a mano)."""
    return montos.assign(id=montos["id"].astype(str)).set_index("id").loc[el["id"]].reset_index()


def _el_con_bandas(n_bandas=10):
    from src.scorecard.scorecard import parametros_scorecard, score_desde_pd

    test, pd_pred, lgd_pred, montos = _entradas()
    el = calcular_el(test, pd_pred, lgd_pred, montos, modelo_pd="mlp", modelo_lgd="mlp")
    el["score"] = score_desde_pd(el["pd"].to_numpy(), parametros_scorecard(SCORECARD))
    el["banda"] = bandas_de_puntaje(el["score"], n_bandas)
    return el, _alinear(montos, el)


def test_calcular_el_une_por_id_y_usa_ead_de_originacion():
    test, pd_pred, lgd_pred, montos = _entradas()
    el = calcular_el(test, pd_pred, lgd_pred, montos, modelo_pd="mlp", modelo_lgd="mlp")
    assert len(el) == len(test) and el["id"].tolist() == test["id"].tolist()
    fila = el.iloc[7]
    lgd_7 = lgd_pred.set_index("id").loc["7", "pred_mlp"]
    assert fila["lgd"] == pytest.approx(lgd_7)
    assert fila["ead"] == pytest.approx(montos.set_index("id").loc[7, "funded_amnt"])  # montos barajado
    assert el["ead"].tolist() == pytest.approx(_alinear(montos, el)["funded_amnt"].tolist())
    assert fila["el"] == pytest.approx(fila["pd"] * fila["lgd"] * fila["ead"])
    assert {"perdida_observada", "perdida_provisional", "principal_no_registrado", "estado"} <= set(el.columns)


def test_calcular_el_falla_si_falta_un_credito():
    test, pd_pred, lgd_pred, montos = _entradas()
    with pytest.raises(ValueError, match="sin predicción"):
        calcular_el(test, pd_pred.iloc[:-1], lgd_pred, montos, modelo_pd="mlp", modelo_lgd="mlp")


def test_resumen_contrasta_solo_los_resueltos():
    el, montos = _el_con_bandas()
    r = resumen_el(el)
    castigado = montos["loan_status"] == "Charged Off"
    mora = montos["loan_status"] == "Late (31-120 days)"
    resuelto = ~mora
    # E[L] de todo el test se sigue reportando
    assert r["total"]["n"] == len(el) and r["total"]["el"] == pytest.approx(el["el"].sum())
    # contraste: E[L] y pérdida realizada de los resueltos (castigados + pagados con pérdida 0)
    pendiente = montos["funded_amnt"] - montos["total_rec_prncp"]
    realizada = (pendiente - 450.0)[castigado].clip(lower=0).sum()
    assert r["resueltos"]["n"] == int(resuelto.sum())
    assert r["resueltos"]["el"] == pytest.approx(el.loc[resuelto.to_numpy(), "el"].sum())
    assert r["resueltos"]["perdida_observada"] == pytest.approx(realizada)
    assert r["resueltos"]["razon_el_sobre_perdida_observada"] == pytest.approx(r["resueltos"]["el"] / realizada)
    # en mora: aparte, con la pérdida provisional (LGD = 1 sobre el pendiente: aquí el 50 % del monto)
    assert r["en_mora"]["n"] == int(mora.sum())
    assert r["en_mora"]["perdida_provisional"] == pytest.approx(0.5 * montos.loc[mora, "funded_amnt"].sum())
    # anomalía de los pagados: n y principal sin registrar (10 USD cada uno), sin sumarlo a la pérdida
    anomalos = (montos["loan_status"] == "Fully Paid") & (pendiente > 0)
    assert r["anomalia_pagados"]["n"] == int(anomalos.sum()) > 0
    assert r["anomalia_pagados"]["principal_no_registrado"] == pytest.approx(10.0 * anomalos.sum())
    # media de (funded − total_rec_prncp)/funded SOLO en los castigados (amortizaron 30 %), no en los en mora (50 %)
    assert r["fraccion_principal_pendiente_media_castigados"] == pytest.approx(0.7)


def test_resumen_por_banda_suma_los_totales():
    el, _ = _el_con_bandas()
    r = resumen_el(el)
    bandas = r["por_banda"]
    assert [b["banda"] for b in bandas] == list(range(1, 11))
    for bloque, claves in (("total", ("n", "el")), ("resueltos", ("n", "el", "perdida_observada")),
                           ("en_mora", ("n", "el", "perdida_provisional"))):
        for clave in claves:
            assert sum(b[bloque][clave] for b in bandas) == pytest.approx(r[bloque][clave]), (bloque, clave)
    for b in bandas:
        res = b["resueltos"]
        assert res["razon_el_sobre_perdida_observada"] == pytest.approx(res["el"] / res["perdida_observada"])
        dentro = el[el["banda"] == b["banda"]]
        assert b["score_min"] == pytest.approx(dentro["score"].min())
        assert b["score_max"] == pytest.approx(dentro["score"].max())


def test_resumen_agrupa_por_la_columna_banda():
    """Las bandas del resumen son las de `el["banda"]`, sean cuales sean (aquí, mitades por posición, no por
    puntaje)."""
    el, _ = _el_con_bandas()
    el["banda"] = np.where(np.arange(len(el)) < 500, 1, 2)
    r = resumen_el(el)
    assert [b["banda"] for b in r["por_banda"]] == [1, 2]
    assert [b["total"]["n"] for b in r["por_banda"]] == [500, len(el) - 500]
    assert r["por_banda"][0]["total"]["el"] == pytest.approx(el["el"].iloc[:500].sum())
    assert r["por_banda"][0]["score_min"] == pytest.approx(el["score"].iloc[:500].min())


def test_razon_sin_perdida_observada_es_none():
    """Sin pérdida realizada la razón no está definida (no se divide por cero)."""
    el = pd.DataFrame({"target": [0, 0], "pd": [0.1, 0.2], "lgd": [0.5, 0.5], "ead": [100.0, 100.0],
                       "el": [5.0, 10.0], "ead_incumplimiento": [0.0, 0.0], "perdida_observada": [0.0, 0.0],
                       "perdida_provisional": [np.nan, np.nan], "principal_no_registrado": [0.0, 0.0],
                       "estado": ["pagado", "pagado"], "observacion_completa": [True, True], "score": [600.0, 580.0],
                       "banda": [1, 1]})
    r = resumen_el(el)
    assert r["resueltos"]["razon_el_sobre_perdida_observada"] is None
    assert r["por_banda"][0]["resueltos"]["razon_el_sobre_perdida_observada"] is None
    assert r["en_mora"]["n"] == 0 and r["en_mora"]["perdida_provisional"] == 0.0
    assert r["fraccion_principal_pendiente_media_castigados"] is None  # sin castigados no hay media


# ---------------------------------------------------------------- comando y config

def test_comando_el_avisa_si_faltan_las_entregas(tmp_path, capsys, monkeypatch):
    """Sin las predicciones de B y D el comando termina con 2, dice qué falta y no escribe nada."""
    from src.common.config import load_config
    from src.el.experimento import manejador

    monkeypatch.chdir(RAIZ)
    cfg = load_config(RAIZ / "configs" / "el" / "base.yaml")
    cfg["pd"]["predicciones"] = str(tmp_path / "no_existe_pd.parquet")
    cfg["lgd"]["predicciones"] = str(tmp_path / "no_existe_lgd.parquet")
    cfg["salida"] = str(tmp_path / "salida")
    assert manejador(cfg) == 2
    salida = capsys.readouterr().out
    assert "Parte B" in salida and "Parte D" in salida
    assert not (tmp_path / "salida").exists()


def _resumen_para_texto(razon):
    base = {"n": 0, "el": 0.0, "ead": 0.0, "tasa_malos": None, "pd_media": None, "lgd_media": None}
    return {"total": {**base, "n": 1234, "el": 5678.4, "ead": 98765.0},
            "resueltos": {**base, "n": 1000, "el": 4000.0, "perdida_observada": 3200.0,
                          "razon_el_sobre_perdida_observada": razon},
            "en_mora": {**base, "n": 234, "el": 1678.4, "perdida_provisional": 9000.0},
            "anomalia_pagados": {"descripcion": "", "n": 7, "principal_no_registrado": 1500.0}}


def test_texto_resumen_totales_resueltos_mora_y_anomalia():
    """Lo comparten `run.py el` y el notebook 05: una línea por bloque, montos en USD sin decimales."""
    from src.el.experimento import texto_resumen

    lineas = texto_resumen(_resumen_para_texto(1.25)).splitlines()
    assert len(lineas) == 4
    todo, resueltos, mora, anomalia = lineas
    assert todo.startswith("Todo el test: 1,234 créditos")
    assert "EAD 98,765 USD" in todo and "E[L] 5,678 USD" in todo
    assert "(n = 1,000)" in resueltos and "E[L] 4,000 USD" in resueltos
    assert "pérdida realizada 3,200 USD" in resueltos and resueltos.endswith("razón 1.250")
    assert "(n = 234)" in mora and "E[L] 1,678 USD" in mora and "LGD = 1" in mora and "9,000 USD" in mora
    assert "7 pagados" in anomalia and "1,500 USD" in anomalia and "no se suma a la pérdida" in anomalia


def test_texto_resumen_sin_perdida_realizada_no_falla():
    """Razón None (sin pérdida realizada): se muestra "—" en vez de fallar al formatear."""
    from src.el.experimento import texto_resumen

    assert texto_resumen(_resumen_para_texto(None)).splitlines()[1].endswith("razón —")


def test_comando_el_imprime_el_texto_resumen(tmp_path, capsys, monkeypatch):
    """`run.py el` imprime exactamente `texto_resumen` del resumen que escribe."""
    import json

    from src.el.experimento import manejador, texto_resumen

    monkeypatch.chdir(tmp_path)
    rutas, _ = _escribir_entradas(tmp_path)
    cfg = _cfg(rutas, tmp_path)
    assert manejador(cfg) == 0
    resumen = json.loads((cfg["salida"] / "summary.json").read_text(encoding="utf-8"))
    assert texto_resumen(resumen) in capsys.readouterr().out


SCORECARD = {"base": 600, "odds_base": 50, "pdo": 20}


def _cfg(rutas, tmp_path, scorecard=SCORECARD, salida="el"):
    """Config mínima de `correr_el`; desde E5 lleva el bloque `scorecard` (lo hereda configs/el/base.yaml)."""
    return {"datos": {"loans": rutas["loans"], "montos": rutas["montos"]},
            "pd": {"predicciones": rutas["pd"], "modelo": "mlp"},
            "lgd": {"predicciones": rutas["lgd"], "modelo": "mlp"},
            "bandas": {"n": 10}, "scorecard": dict(scorecard), "salida": tmp_path / salida}


def _escribir_entradas(tmp_path):
    """Escribe las entradas; `loans` lleva también filas de train (la huella de la partición las incluye)."""
    test, pd_pred, lgd_pred, montos = _entradas()
    train = pd.DataFrame({"id": ["t1", "t2"], "target": [0, 1], "split": "train"}).astype({"target": "int8"})
    loans = pd.concat([train, test.assign(split="test")], ignore_index=True)
    rutas = {n: tmp_path / f"{n}.parquet" for n in ("loans", "montos", "pd", "lgd")}
    loans.to_parquet(rutas["loans"]); montos.to_parquet(rutas["montos"])
    pd_pred.to_parquet(rutas["pd"]); lgd_pred.to_parquet(rutas["lgd"])
    return rutas, loans


def test_config_el_hereda_la_scorecard_y_usa_la_logistica():
    """configs/el/base.yaml hereda el bloque scorecard de la Parte C (así `--set scorecard.pdo=40` funciona), usa la
    PD de la logística (la única entregada por B hasta B5) y la LGD del MLP, con deciles del puntaje."""
    from src.common.config import apply_overrides, load_config

    cfg = load_config(RAIZ / "configs" / "el" / "base.yaml")
    assert cfg["scorecard"] == load_config(RAIZ / "configs" / "scorecard" / "base.yaml")["scorecard"]
    assert cfg["pd"]["modelo"] == "logistica" and cfg["lgd"]["modelo"] == "mlp"
    assert cfg["bandas"]["n"] == 10
    assert apply_overrides(cfg, ["scorecard.pdo=40"])["scorecard"]["pdo"] == 40


def test_bandas_de_puntaje_numeradas_de_menor_a_mayor_puntaje():
    score = pd.Series([700.0, 500.0, 650.0, 550.0, 600.0, 520.0, 680.0, 580.0, 620.0, 640.0], index=range(10, 20))
    banda = bandas_de_puntaje(score, n_bandas=5)
    assert banda.index.tolist() == score.index.tolist()  # se alinea con el crédito, no con la posición
    assert sorted(banda.unique().tolist()) == [1, 2, 3, 4, 5]
    assert banda[11] == 1 and banda[15] == 1  # los dos menores puntajes (500 y 520): banda 1, más riesgo
    assert banda[10] == 5 and banda[16] == 5  # los dos mayores (700 y 680): banda 5
    for b in range(1, 5):  # cada banda queda por debajo de la siguiente
        assert score[banda == b].max() < score[banda == b + 1].min()


def test_bandas_de_puntaje_con_todos_los_puntajes_iguales_es_error():
    with pytest.raises(ValueError, match="iguales"):
        bandas_de_puntaje(pd.Series([550.0] * 20), n_bandas=10)


def test_bandas_de_puntaje_con_empates_da_menos_bandas_numeradas_desde_1():
    """Con empates `qcut` fusiona bordes repetidos: salen k < n bandas, numeradas 1…k."""
    score = pd.Series([500.0] * 8 + [600.0, 700.0])
    banda = bandas_de_puntaje(score, n_bandas=5)
    assert sorted(banda.unique().tolist()) == list(range(1, banda.nunique() + 1)) and banda.nunique() < 5
    assert banda[score == 500.0].nunique() == 1 and banda[score == 500.0].iloc[0] == 1


def test_n_bandas_pedidas_y_obtenidas_y_criterio(tmp_path):
    """El criterio de summary.json dice cuántas bandas se pidieron; con empates en la PD se obtienen menos."""
    from src.el.experimento import correr_el

    rutas, _ = _escribir_entradas(tmp_path)
    cfg = _cfg(rutas, tmp_path)
    cfg["bandas"]["n"] = 5
    r = correr_el(cfg, Path("/"))
    assert r["n_bandas_pedidas"] == 5 and r["n_bandas_obtenidas"] == len(r["por_banda"]) == 5
    assert "5 bandas" in r["bandas"] and "deciles" not in r["bandas"]
    # empates: el 80 % de los créditos con la misma PD (mismo puntaje) → se fusionan bandas
    pd_pred = pd.read_parquet(rutas["pd"])
    pd_pred.loc[pd_pred.index[: int(0.8 * len(pd_pred))], "pred_mlp"] = 0.3
    pd_pred.to_parquet(rutas["pd"])
    r = correr_el(_cfg(rutas, tmp_path, salida="empates"), Path("/"))
    assert r["n_bandas_pedidas"] == 10 and r["n_bandas_obtenidas"] == len(r["por_banda"]) < 10


def test_descripcion_de_la_anomalia_nombra_los_dos_estados_pagados():
    el, _ = _el_con_bandas()
    descripcion = resumen_el(el)["anomalia_pagados"]["descripcion"]
    assert all(estado in descripcion for estado in PAGADOS)


def test_correr_el_de_punta_a_punta(tmp_path):
    from src.el.experimento import correr_el
    from src.scorecard.scorecard import parametros_scorecard, score_desde_pd

    rutas, loans = _escribir_entradas(tmp_path)
    r = correr_el(_cfg(rutas, tmp_path), Path("/"))
    assert (tmp_path / "el" / "summary.json").is_file()
    el = pd.read_parquet(tmp_path / "el" / "el_test.parquet")
    assert r["total"]["n"] == len(el) == int((loans["split"] == "test").sum())
    # contrato del núcleo §5.4: las columnas listadas no se quitan ni se renombran
    assert {"id", "target", "pd", "lgd", "ead", "el", "perdida_observada"} <= set(el.columns)
    params = parametros_scorecard(SCORECARD)
    assert el["score"].to_numpy() == pytest.approx(score_desde_pd(el["pd"].to_numpy(), params), abs=1e-9)
    assert r["scorecard"] == pytest.approx(params)  # base, odds_base, pdo, factor, offset
    bandas = r["por_banda"]
    assert [b["banda"] for b in bandas] == list(range(1, 11))
    for menor, mayor in zip(bandas, bandas[1:]):  # banda 1 = menor puntaje = mayor PD
        assert menor["score_max"] < mayor["score_min"]
        assert menor["total"]["pd_media"] > mayor["total"]["pd_media"]


def test_correr_el_escribe_la_evidencia(tmp_path):
    """Núcleo §6, "Mismos datos": huella de la partición (todas las filas de loans, en su orden) y de los dos archivos
    de predicciones que consume la E[L]."""
    from src.el.experimento import correr_el
    from src.pd.experimento import huella_archivo, huella_particion

    rutas, loans = _escribir_entradas(tmp_path)
    r = correr_el(_cfg(rutas, tmp_path), Path("/"))
    assert r["evidencia"] == {"sha256_particion": huella_particion(loans["id"], loans["split"]),
                              "sha256_pred_pd": huella_archivo(rutas["pd"]),
                              "sha256_pred_lgd": huella_archivo(rutas["lgd"])}


def test_cambiar_el_pdo_cambia_los_rangos_de_puntaje_pero_no_los_creditos_de_cada_banda(tmp_path):
    """El puntaje es una transformación creciente de −logit(PD): con otro PDO cambian los rangos de cada decil, pero
    los deciles agrupan los mismos créditos y la E[L] por banda no cambia (lo que muestra la sustentación)."""
    from src.el.experimento import correr_el

    rutas, _ = _escribir_entradas(tmp_path)
    r20 = correr_el(_cfg(rutas, tmp_path, salida="pdo20"), Path("/"))
    r40 = correr_el(_cfg(rutas, tmp_path, scorecard={**SCORECARD, "pdo": 40}, salida="pdo40"), Path("/"))
    assert r40["scorecard"]["pdo"] == 40 and r40["scorecard"]["factor"] == pytest.approx(2 * r20["scorecard"]["factor"])
    assert [b["score_min"] for b in r20["por_banda"]] != pytest.approx([b["score_min"] for b in r40["por_banda"]])
    assert [b["total"]["el"] for b in r20["por_banda"]] == pytest.approx([b["total"]["el"] for b in r40["por_banda"]])
