"""Implementación de referencia del verificador con mutantes por variable de entorno MUT."""
import math, os
import numpy as np

MUT = os.environ.get("MUT", "")
CLAVES_SCORECARD = ("base", "odds_base", "pdo")
CLAVES_PARAMETROS = ("base", "odds_base", "pdo", "factor", "offset")

def _real(v, nombre):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
        raise ValueError(nombre)
    v = float(v)
    if not math.isfinite(v):
        raise ValueError(nombre)
    return v

def validar_cfg_scorecard(cfg):
    if not isinstance(cfg, dict) or set(cfg) != set(CLAVES_SCORECARD):
        raise ValueError("cfg")
    r = {k: _real(cfg[k], k) for k in CLAVES_SCORECARD}
    if r["pdo"] <= 0 or r["odds_base"] <= 0:
        raise ValueError("pos")
    return r

def bloque_scorecard(cfg):
    if not isinstance(cfg, dict) or "scorecard" not in cfg:
        raise ValueError("bloque")
    return validar_cfg_scorecard(cfg["scorecard"])

def parametros_scorecard(cfg):
    c = validar_cfg_scorecard(cfg)
    ln = math.log
    if MUT == "log10_factor":
        factor = c["pdo"] / math.log10(2)
    elif MUT == "log10_ambos":
        factor = c["pdo"] / math.log10(2)
    elif MUT == "factor_por":
        factor = c["pdo"] * math.log(2)
    else:
        factor = c["pdo"] / math.log(2)
    if MUT in ("log10_offset", "log10_ambos"):
        offset = c["base"] - factor * math.log10(c["odds_base"])
    elif MUT == "offset_mas":
        offset = c["base"] + factor * math.log(c["odds_base"])
    else:
        offset = c["base"] - factor * math.log(c["odds_base"])
    if MUT == "signo_factor":
        factor = -factor
        offset = c["base"] - factor * math.log(c["odds_base"])
    return {**c, "factor": float(factor), "offset": float(offset)}

def _params(params):
    if not isinstance(params, dict):
        raise ValueError("params")
    if MUT == "sin_claves_exactas":
        if not set(CLAVES_PARAMETROS) <= set(params):
            raise ValueError("params")
    elif set(params) != set(CLAVES_PARAMETROS):
        raise ValueError("params")
    r = {k: _real(params[k], k) if MUT != "bool_params" else float(params[k]) for k in CLAVES_PARAMETROS}
    if MUT == "bool_params" and not all(math.isfinite(r[k]) for k in r):
        raise ValueError("nf")
    if MUT != "signo_factor" and r["factor"] <= 0:
        raise ValueError("factor")
    if MUT == "signo_factor" and r["factor"] == 0:
        raise ValueError("factor")
    if MUT == "recalcula":
        c = parametros_scorecard({k: r[k] for k in CLAVES_SCORECARD})
        r["factor"], r["offset"] = c["factor"], c["offset"]
    return r

def _salida(x):
    x = np.asarray(x, dtype=np.float64)
    if MUT == "redondea":
        x = np.round(x)
    return np.float64(x) if x.ndim == 0 else x

def _finito(x):
    a = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError("no finito")
    return a

def score_desde_logit(z, params):
    p = _params(params); z = _finito(z)
    if MUT == "odds_malo_bueno":
        return _salida(p["offset"] + p["factor"] * z)
    return _salida(p["offset"] - p["factor"] * z)

def score_desde_pd(pd, params):
    pr = _params(params)
    a = np.asarray(pd, dtype=np.float64)
    if MUT == "acepta_0_1":
        if not np.all((a >= 0) & (a <= 1)):
            raise ValueError("p")
        with np.errstate(divide="ignore"):
            z = np.log(a) - np.log1p(-a)
        return _salida(pr["offset"] - pr["factor"] * z)
    if not np.all((a > 0) & (a < 1)):
        raise ValueError("p")
    if MUT == "lineal_p":
        # "PDO puntos por cada 1 % de PD", anclado en p_base
        pb = 1 / (1 + pr["odds_base"])
        return _salida(pr["base"] - pr["pdo"] * (a - pb) * 100)
    if MUT == "log2_logit":
        z = np.log2(a) - np.log2(1 - a)
    else:
        z = np.log(a) - np.log1p(-a)
    return score_desde_logit(z, params)

def logit_desde_score(s, params):
    p = _params(params); s = _finito(s)
    if MUT == "odds_malo_bueno":
        return _salida((s - p["offset"]) / p["factor"])
    return _salida((p["offset"] - s) / p["factor"])

def pd_desde_score(s, params):
    z = np.asarray(logit_desde_score(s, params), dtype=np.float64)
    out = np.where(z >= 0, 1 / (1 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1 + np.exp(-np.abs(z))))
    return _salida(out)

# ---- mutantes adicionales ----
_sdl, _lds, _pds, _sdp = score_desde_logit, logit_desde_score, pd_desde_score, score_desde_pd
if MUT == "recalcula_solo_pd":
    def score_desde_pd(pd, params):
        r = _params(params)
        c = parametros_scorecard({k: r[k] for k in CLAVES_SCORECARD})
        return _sdp(pd, c)
    def pd_desde_score(s, params):
        r = _params(params)
        c = parametros_scorecard({k: r[k] for k in CLAVES_SCORECARD})
        return _pds(s, c)
if MUT == "pd_complemento":
    def pd_desde_score(s, params):
        return _salida(1 - np.asarray(_pds(s, params)))
if MUT == "sigmoide_ingenua":
    def pd_desde_score(s, params):
        z = np.asarray(logit_desde_score(s, params))
        return _salida(1 / (1 + np.exp(-z)))
if MUT == "logit_ingenuo":
    def score_desde_pd(pd, params):
        a = np.asarray(pd, dtype=np.float64)
        if not np.all((a > 0) & (a < 1)): raise ValueError("p")
        return _sdl(np.log(a / (1 - a)), params)
if MUT == "float32_salida":
    def score_desde_logit(z, params):
        r = _sdl(z, params)
        return _salida(np.asarray(r, dtype=np.float32).astype(np.float64))
if MUT == "pd_sin_validar_rango_array":
    def score_desde_pd(pd, params):
        a = np.asarray(pd, dtype=np.float64)
        if a.ndim == 0: return _sdp(pd, params)
        with np.errstate(all="ignore"):
            return _sdl(np.log(a) - np.log1p(-a), params) if np.all(np.isfinite(np.log(a)-np.log1p(-a))) else _sdl(np.log(a) - np.log1p(-a), params)
if MUT == "odds_base_no_validado":
    def validar_cfg_scorecard(cfg):
        if not isinstance(cfg, dict) or set(cfg) != set(CLAVES_SCORECARD):
            raise ValueError("cfg")
        r = {k: _real(cfg[k], k) for k in CLAVES_SCORECARD}
        if r["pdo"] <= 0: raise ValueError("pos")
        return r
if MUT == "pdo_negativo_aceptado":
    def validar_cfg_scorecard(cfg):
        if not isinstance(cfg, dict) or set(cfg) != set(CLAVES_SCORECARD):
            raise ValueError("cfg")
        r = {k: _real(cfg[k], k) for k in CLAVES_SCORECARD}
        if r["pdo"] == 0 or r["odds_base"] <= 0: raise ValueError("pos")
        return r
if MUT == "params_alias_mutable":
    _pp = parametros_scorecard
    def parametros_scorecard(cfg):
        r = _pp(cfg)
        cfg.update({"factor": r["factor"], "offset": r["offset"]}) if isinstance(cfg, dict) else None
        return r
if MUT == "offset_positivo":
    _pp2 = _params
    def _params(params):
        r = _pp2(params)
        if r["offset"] <= 0: raise ValueError("offset")
        return r
