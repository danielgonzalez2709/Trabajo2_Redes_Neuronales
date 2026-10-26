import json, math, numpy as np, pandas as pd
from scipy.stats import spearmanr
R = "."  # correr desde la raíz del repositorio (rutas de Windows)
el = pd.read_parquet(R + r"\results\data\el\el_test.parquet")
s = json.load(open(R + r"\results\data\el\summary.json", encoding="utf-8"))
sc = s["scorecard"]
logit = np.log(el.pd/(1-el.pd))
print("max |score - (offset - factor*logit)|", np.abs(el.score - (sc["offset"] - sc["factor"]*logit)).max())
print("spearman score-pd", spearmanr(el.score, el.pd).correlation)
b = {x["banda"]: x for x in s["por_banda"]}
for k in ["el", "ead", "pd_media", "lgd_media"]:
    print(k, b[1]["total"][k]/b[10]["total"][k])
real = [b[i]["resueltos"]["perdida_observada"] for i in range(1, 11)]
tasa = [b[i]["total"]["tasa_malos"] for i in range(1, 11)]
print("realizada monotona dec", all(np.diff(real) < 0), [round(x/1e6,2) for x in real])
print("tasa monotona dec", all(np.diff(tasa) < 0))
print("0.9251*0.9696*1.1788 =", 0.9251*0.9696*1.1788)
res = el[el.observacion_completa]
def alt(g):
    c = g[g.estado == "castigado"]
    lo = c.perdida_observada / c.ead_incumplimiento
    fpd = (g.pd*g.lgd*g.ead).sum() / (c.lgd*c.ead).sum()
    flgd = (c.lgd*c.ead).sum() / (lo*c.ead).sum()
    fead = (lo*c.ead).sum() / (lo*c.ead_incumplimiento).sum()
    fead2 = (c.lgd*c.ead).sum() / (c.lgd*c.ead_incumplimiento).sum()
    flgd2 = (c.lgd*c.ead_incumplimiento).sum() / (lo*c.ead_incumplimiento).sum()
    return fpd, flgd, fead, fead2, flgd2, fpd*fead2*flgd2, (g.el.sum()/g.perdida_observada.sum())
print("total", ["%.4f" % x for x in alt(res)])
for k, g in res.groupby("banda"):
    print(k, ["%.4f" % x for x in alt(g)])
