import math, numpy as np, pytest
from src.scorecard.scorecard import score_desde_logit, score_desde_pd, logit_desde_score, pd_desde_score, parametros_scorecard
FUNCIONES = [score_desde_logit, score_desde_pd, logit_desde_score, pd_desde_score]
ENTRADA = {score_desde_logit: 0.0, score_desde_pd: 0.3, logit_desde_score: 600.0, pd_desde_score: 600.0}
def pv():
    f = 20 / math.log(2)
    return {"base": 600.0, "odds_base": 50.0, "pdo": 20.0, "factor": f, "offset": 600 - f * math.log(50)}

@pytest.mark.parametrize("clave", ["factor", "offset"])
@pytest.mark.parametrize("valor", [True, False, np.bool_(True), np.bool_(False)], ids=["True","False","np.True_","np.False_"])
@pytest.mark.parametrize("f", FUNCIONES, ids=lambda f: f.__name__)
def test_bool_en_params(f, clave, valor):
    with pytest.raises(ValueError):
        f(ENTRADA[f], {**pv(), clave: valor})

def test_pd_usa_factor_offset_recibidos():
    params = {"base": 600.0, "odds_base": 50.0, "pdo": 20.0, "factor": 10.0, "offset": 100.0}
    p = 1 / (1 + math.exp(-2.0))  # sigma(2) = 0.8807970779778823
    assert score_desde_pd(p, params) == pytest.approx(80.0, abs=1e-9)
    assert pd_desde_score(80.0, params) == pytest.approx(0.8807970779778823, abs=1e-12)

def test_offset_negativo_valido():
    params = parametros_scorecard({"base": 0, "odds_base": 50, "pdo": 10})
    # offset = 0 - (10/ln2) ln 50 = -10 log2(50) = -56.43856189774724
    assert params["offset"] == pytest.approx(-56.43856189774724, abs=1e-9)
    assert score_desde_logit(0.0, params) == pytest.approx(-56.43856189774724, abs=1e-9)
