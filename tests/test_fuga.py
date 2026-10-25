"""A3 — Fuga de información (núcleo §5.7): ninguna variable posterior a la originación en configs/features.yaml."""

from pathlib import Path

import pytest
import yaml

RAIZ = Path(__file__).resolve().parents[1]
FEATURES = yaml.safe_load((RAIZ / "configs" / "features.yaml").read_text(encoding="utf-8"))["features"]
NOMBRES = [f["nombre"] for f in FEATURES]

# Posteriores a la originación (núcleo §5.7 + diccionario), identificadores, texto libre, puntaje/precio de
# Lending Club y variables sensibles (Parte A §3.4).
PROHIBIDAS = {
    "loan_status", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp",
    "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_pymnt_d",
    "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d", "pymnt_plan",
    "id", "member_id", "url", "desc", "title", "emp_title",
    "grade", "sub_grade", "int_rate", "installment", "funded_amnt", "funded_amnt_inv", "initial_list_status",
    "zip_code", "addr_state",
}
CAMPOS = {"nombre", "tipo", "disponible_en_originacion", "justificacion", "cita_diccionario", "transformacion",
          "en_app", "etiqueta_es", "ayuda_es"}
CAMPOS_CAT = CAMPOS | {"categorias_es"}  # las categóricas traen además la etiqueta en español de cada código (§5.1)


PREFIJOS_PROHIBIDOS = ("total_pymnt", "total_rec_", "out_prncp", "last_pymnt_")  # patrones del núcleo §5.7


def test_ninguna_variable_prohibida():
    assert not set(NOMBRES) & PROHIBIDAS
    assert not [n for n in NOMBRES if n.startswith(PREFIJOS_PROHIBIDOS)]


def test_sin_repetidas_y_cantidad_para_un_formulario():
    assert len(NOMBRES) == len(set(NOMBRES))
    assert 12 <= len(NOMBRES) <= 18


@pytest.mark.parametrize("f", FEATURES, ids=NOMBRES)
def test_cada_variable_cumple_el_contrato(f):
    assert set(f) == (CAMPOS_CAT if f["tipo"] == "cat" else CAMPOS)
    assert f["tipo"] in {"num", "cat"}
    assert f["disponible_en_originacion"] is True
    assert f["transformacion"] in {"ninguna", "log1p"}
    assert f["cita_diccionario"].strip() and f["justificacion"].strip()


def test_cada_categoria_que_ve_la_app_tiene_etiqueta_en_espanol():
    """Todas las categorías del preprocesador (salvo OTRA, interna) tienen etiqueta en categorias_es, y nada sobra."""
    import json

    pre = json.loads((RAIZ / "models" / "preprocesador_pd.json").read_text(encoding="utf-8"))
    etiquetas = {f["nombre"]: f["categorias_es"] for f in FEATURES if f["tipo"] == "cat"}
    for v in pre["variables"]:
        if v["tipo"] == "cat":
            esperadas = set(v["categorias"]) - {"OTRA"}
            assert set(etiquetas[v["nombre"]]) == esperadas, v["nombre"]
            assert all(str(e).strip() for e in etiquetas[v["nombre"]].values())
