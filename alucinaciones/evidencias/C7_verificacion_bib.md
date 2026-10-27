# C7 — verificación bibliográfica (27-oct-2026)

Método: `curl -H "Accept: application/x-bibtex" https://doi.org/<doi>` y `https://api.crossref.org/works/<doi>`.

| clave | fuente | resultado |
|---|---|---|
| breiman2001 | doi.org 10.1023/A:1010933404324 + Crossref | "Random Forests", Mach. Learn. 45(1), 5-32, oct. 2001. Coincide con lo dado. |
| aas2021 | doi.org 10.1016/j.artint.2021.103502 + Crossref | Artif. Intell. 298, art. 103502, sept. 2021; título con subtítulo "More accurate approximations to Shapley values". Coincide. |
| fisher2019 | https://jmlr.org/papers/v20/18-760.bib | JMLR 20(177), 1-81, 2019; título completo "...Learning a Variable's Importance by Studying an Entire Class of Prediction Models Simultaneously". Coincide. |
| janzing2020 | https://proceedings.mlr.press/v108/janzing20a.html (BibTeX y meta citation_*) | PMLR 108, pp. 2907-2916, 2020 (AISTATS, eds. Chiappa y Calandra); autores Janzing, Minorics, Bloebaum. Coincide. |
| strumbelj2014 | doi.org 10.1007/s10115-013-0679-x + Crossref + página de Springer | NO agregada (ver abajo). |

## Discrepancias
- Ninguna en las 4 agregadas.
- Štrumbelj y Kononenko: el encargo dice 2014. Crossref: issued/published-online 2013-08-30, published-print 2014-12 (vol. 41, núm. 3, pp. 647-665); el BibTeX de doi.org da year=2013 (clave `_trumbelj_2013`); Springer: citation_publication_date 2014/12, online 2013/08/30. Ambos años son defendibles; hay que decidir.

## Štrumbelj: ¿muestreo de permutaciones?
Fuente primaria accesible: solo el resumen oficial de Springer (texto completo de pago, "This is a preview of subscription content"):
"We present a sensitivity analysis-based method for explaining prediction models ... all subsets of input features are perturbed, so interactions and redundancies between features are taken into account. ... when explaining an additive model, the method is equivalent to commonly used additive model-specific methods."
El resumen NO menciona Shapley, muestreo ni permutaciones. Lo único con "sampling" en la página es la referencia de Castro et al. (2009). Veredicto: no confirmado en fuente primaria -> no se agrega `strumbelj2014`.

## mkdocs
Ver resultado en el informe de la tarea (`mkdocs build --strict`).
