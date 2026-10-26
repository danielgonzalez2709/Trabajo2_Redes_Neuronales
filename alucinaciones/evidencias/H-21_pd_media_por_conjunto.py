import os, sys
R="."; sys.path.insert(0,R)  # correr desde la raíz del repositorio
import numpy as np, pandas as pd
from src.pd.experimento import RUTAS_PD, cargar_matrices
from src.pd.mlp import MLPPD
from src.pd.logistica import LogisticaPD
M=cargar_matrices(RUTAS_PD)
lg=LogisticaPD.cargar('models/pd_logistica.joblib'); ml=MLPPD.cargar('models/pd_mlp.pt')
pt=pd.read_parquet('results/data/pd/pred_test.parquet')
for s,(X,y,ids) in M.items():
    pl=lg.predict_proba(X); pm=ml.predict_proba(X)
    print(s, len(y), 'tasa %.4f'%np.mean(y), 'log %.4f'%pl.mean(), 'mlp %.4f'%pm.mean())
    if s=='test':
        d=pd.DataFrame({'id':ids,'pl':pl,'pm':pm}).merge(pt,on='id')
        print(' max diff vs pred_test', np.abs(d.pl-d.pred_logistica).max(), np.abs(d.pm-d.pred_mlp).max())
        # top decile calibration
        for col in ['pred_logistica','pred_mlp']:
            q=d[col]>=d[col].quantile(0.9); print(' ',col,'top10%% mean pd %.3f tasa %.3f'%(d.loc[q,col].mean(), d.loc[q,'y'].mean()))
