import json, numpy as np, pandas as pd
R = "."  # correr desde la raíz del repositorio (rutas de Windows)
el = pd.read_parquet(R + r"\results\data\el\el_test.parquet")
print(el.columns.tolist(), len(el))
loans = pd.read_parquet(R + r"\data\processed\loans_pd.parquet")
montos = pd.read_parquet(R + r"\data\processed\montos.parquet")
print("loans", loans.shape, "montos", montos.shape, montos.columns.tolist())
print("same id order:", (loans["id"].astype(str).values == montos["id"].astype(str).values).all())
print(loans.columns.tolist())
test = loans[loans.split == "test"]
mt = montos[(loans.split == "test").values]
print("test n", len(test))
if "issue_d" in loans.columns: print(loans.loc[loans.split=="test","issue_d"].astype(str).str[:4].value_counts() if loans["issue_d"].dtype==object else loans.loc[loans.split=="test","issue_d"].dt.year.value_counts())
print(mt.loan_status.value_counts())
for s in ["train","val","test"]:
    print(s, loans.loc[loans.split==s,"target"].mean())
# anomalia
fp = mt[mt.loan_status.isin(["Fully Paid","Does not meet the credit policy. Status:Fully Paid"])]
dif = fp.funded_amnt - fp.total_rec_prncp
an = dif[dif > 0]
print("anom n", len(an), "sum", round(an.sum(),2), "n>1", (an>1).sum(), "sum>1", round(an[an>1].sum(),2), "median", an.median())
# castigados
ch = mt[mt.loan_status.isin(["Charged Off","Does not meet the credit policy. Status:Charged Off"])]
pend = ch.funded_amnt - ch.total_rec_prncp
print("castigados", len(ch), "min pend", pend.min(), "frac pend media", (pend/ch.funded_amnt).mean())
print("net rec negative?", ((ch.recoveries-ch.collection_recovery_fee)<0).sum(), "P<0", (montos.total_rec_prncp<0).sum())
mo = mt[mt.loan_status.isin(["Late (31-120 days)","Default"])]
print("mora", len(mo), "rec>0", (mo.recoveries>0).sum(), "fee>0", (mo.collection_recovery_fee>0).sum(), "prov", (mo.funded_amnt-mo.total_rec_prncp).clip(lower=0).sum())
# LGD pred
lp = pd.read_parquet(R + r"\results\data\lgd\pred_pd_test.parquet")
print("lgd pred", lp.shape, lp.columns.tolist(), "std pop", lp.pred_mlp.std(ddof=0), "std m", lp.pred_mlp.std(), lp.pred_mlp.min(), lp.pred_mlp.max(), lp.pred_mlp.mean())
# L_i equals LGDobs*(F-P)
lgdobs = np.clip(1 - (ch.recoveries-ch.collection_recovery_fee)/pend, 0, 1)
L = np.maximum(0, pend - (ch.recoveries-ch.collection_recovery_fee))
print("max |L - lgdobs*pend|", np.abs(L - lgdobs*pend).max())
# decomposition
res = el[el.observacion_completa]
def dec(g):
    c = g[g.estado=="castigado"]
    lo = c.perdida_observada / c.ead_incumplimiento
    num = (g.pd*g.lgd*g.ead).sum()
    fpd = num / (c.lgd*c.ead).sum()
    flgd = (c.lgd*c.ead).sum() / (lo*c.ead).sum()
    fead = (lo*c.ead).sum() / (lo*c.ead_incumplimiento).sum()
    return num/g.perdida_observada.sum(), fpd, flgd, fead
for b, g in res.groupby("banda"):
    print(b, ["%.3f"%x for x in dec(g)])
print("T", ["%.3f"%x for x in dec(res)], dec(res))
print("score monotone in pd?", np.corrcoef(el.score.rank(), el.pd.rank())[0,1])
print("score range", el.score.min(), el.score.max())
# resueltos estados in test
print(res.estado.value_counts())
# H-14: razón banda 1 / banda 10 de E[L], EAD (suma) y PD media; reparto en escala logarítmica
b1, b10 = el[el.banda == 1], el[el.banda == 10]
r = pd.Series({"ead": b1.ead.sum() / b10.ead.sum(), "pd": b1.pd.mean() / b10.pd.mean(), "el": b1.el.sum() / b10.el.sum()})
print(r)
print("share log PD", np.log(r["pd"]) / np.log(r["el"]), "share log EAD", np.log(r["ead"]) / np.log(r["el"]))
print(0.925 * 0.970 * 1.179)  # producto de los factores redondeados que escribió el redactor
