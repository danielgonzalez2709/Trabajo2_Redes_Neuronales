"""H-13: la pérdida observada de la 1.ª versión de E5 (24-oct) aplicada a todos los estados de test.

Fórmula de la 1.ª versión: max(0, (funded_amnt − total_rec_prncp) − (recoveries − collection_recovery_fee)) para todo crédito.
Correr desde la raíz del repositorio: .venv\\Scripts\\python alucinaciones\\evidencias\\H-13_perdida_observada.py
"""
import pandas as pd

loans = pd.read_parquet("data/processed/loans_pd.parquet", columns=["id", "split"])
montos = pd.read_parquet("data/processed/montos.parquet")
test = montos[montos["id"].isin(loans.loc[loans["split"] == "test", "id"])]
pendiente = test["funded_amnt"] - test["total_rec_prncp"]
recuperado = test["recoveries"] - test["collection_recovery_fee"]
perdida_v1 = (pendiente - recuperado).clip(lower=0)

pagados = test["loan_status"].str.contains("Fully Paid")
mora = test["loan_status"].isin(["Late (31-120 days)", "Default"])
castigados = test["loan_status"].str.contains("Charged Off")
print(f"créditos de test: {len(test):,}")
print(f"pérdida 'observada' v1 total: {perdida_v1.sum():,.2f} USD")
print(f"  castigados (pérdida realizada): {perdida_v1[castigados].sum():,.2f} USD")
print(f"  Fully Paid con pérdida > 0: {(perdida_v1[pagados] > 0).sum():,} créditos, {perdida_v1[pagados].sum():,.2f} USD "
      f"({(perdida_v1[pagados] > 1).sum():,} con más de 1 USD)")
print(f"  en mora: {mora.sum():,} créditos, {perdida_v1[mora].sum():,.2f} USD "
      f"= {perdida_v1[mora].sum() / perdida_v1.sum():.1%} del total; recoveries > 0 en mora: {(test.loc[mora, 'recoveries'] > 0).sum()}")
