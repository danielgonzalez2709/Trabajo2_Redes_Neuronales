import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np, json
from src.lgd.objetivo import resumen_objetivo, calcular_lgd, es_poblacion_lgd
m=pd.read_parquet('data/processed/montos.parquet'); p=pd.read_parquet('data/processed/loans_pd.parquet')
print(m.shape,p.shape, (m.id.values==p.id.values).all()); print(p.columns.tolist())
print(m.loan_status.value_counts())
print(json.dumps(resumen_objetivo(m),indent=1,ensure_ascii=False))
pob=es_poblacion_lgd(m.loan_status)
r=calcular_lgd(m[pob]); d=m[pob].join(r); d['issue_d']=p.loc[pob,'issue_d'].values; d['split']=p.loc[pob,'split'].values
v=d[d.lgd_valida]
print('lgd==1',(v.lgd==1).sum(),'rec==0',(v.recoveries==0).sum(), 'both',((v.lgd==1)&(v.recoveries==0)).sum())
print('lgd==1 & rec>0',((v.lgd==1)&(v.recoveries>0)).sum())
d['anio']=pd.to_datetime(d.issue_d).dt.year
g=d.groupby('split').agg(n=('lgd','size'),p_lgd1=('lgd',lambda s:(s==1).mean()),p_rec0=('recoveries',lambda s:(s==0).mean()),med=('lgd','median'))
print(g)
g=d.groupby('anio').agg(n=('lgd','size'),p_rec0=('recoveries',lambda s:(s==0).mean()),p_lgd1=('lgd',lambda s:(s==1).mean()),lgd_media=('lgd','mean'),lgd_med_rec_pos=('lgd',lambda s: s[d.loc[s.index,'recoveries']>0].mean()), ead_media=('ead','mean'), frac_ead=('ead',lambda s:(s/d.loc[s.index,'funded_amnt']).mean()))
print(g.to_string())
# by quarter
d['q']=pd.to_datetime(d.issue_d).dt.to_period('Q')
print(d.groupby('q').agg(n=('lgd','size'),p_rec0=('recoveries',lambda s:(s==0).mean())).to_string())
# fee ratio
x=d[d.recoveries>0]; q=(x.collection_recovery_fee/x.recoveries)
print(q.describe()); print(q.round(3).value_counts().head(15))
print('fee>rec',(d.collection_recovery_fee>d.recoveries).sum(),'rec0fee>0',((d.recoveries==0)&(d.collection_recovery_fee>0)).sum())
print('neg lgd examples'); print(v[v.lgd_sin_recorte<0][['funded_amnt','total_rec_prncp','recoveries','collection_recovery_fee','ead','lgd_sin_recorte']].head(10))
# fee ratio by year
x=x.assign(r=q); print(x.groupby('anio').r.agg(['mean','median',lambda s:(s.round(2)==0.18).mean(),lambda s:(s.round(2)==0.01).mean()]))
# composition: grade by year among pob
print(pd.crosstab(d.anio, p.loc[pob,'term'].values if 'term' in p else 0, normalize='index'))
print('=== test interno de censura: p_rec0 por año x quintil de fracción de principal pagado (1-EAD/funded) ===')
d['pagado']=1-d.ead/d.funded_amnt
d['term']=p.loc[pob,'term'].values
d['qp']=pd.cut(d.pagado,[-0.01,0.1,0.25,0.5,0.75,1.0])
print(pd.pivot_table(d,index='anio',columns='qp',values='recoveries',aggfunc=lambda s:(s==0).mean(),observed=False).round(3).to_string())
print(pd.pivot_table(d,index='anio',columns='qp',values='recoveries',aggfunc='size',observed=False).to_string())
print(pd.pivot_table(d,index='anio',columns='term',values='recoveries',aggfunc=lambda s:(s==0).mean()).round(3))
print(pd.crosstab(d.anio,d.loan_status))
print(d[d.anio<=2009].groupby('loan_status').recoveries.apply(lambda s:(s==0).mean()))
