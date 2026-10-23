"""A8 — Verificación de las referencias candidatas del Trabajo 2 contra doi.org y Crossref (23-oct-2026).

Los DOI de CANDIDATAS los propuso Claude (Opus 5.5) de memoria al planear las partes B, C y D; aquí se comprueba cada uno.
Para cada DOI: ¿resuelve en doi.org? ¿coincide el título/autor/año esperado? Se imprime el BibTeX oficial.
Uso (con internet): python alucinaciones/evidencias/A8_verificacion_bib.py > alucinaciones/evidencias/A8_verificacion_bib.txt
"""
import json
import sys
import urllib.error
import urllib.request

# clave: (DOI propuesto de memoria, lo que se espera encontrar)
CANDIDATAS = {
    "akiba2019": ("10.1145/3292500.3330701", "Optuna: a next-generation hyperparameter optimization framework"),
    "rumelhart1986": ("10.1038/323533a0", "Learning representations by back-propagating errors"),
    "brier1950": ("10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2", "Verification of forecasts expressed in terms of probability"),
    "papke1996": ("10.1002/(SICI)1099-1255(199611)11:6<619::AID-JAE418>3.0.CO;2-1", "Econometric methods for fractional response variables"),
    "kingma2015": ("10.48550/arXiv.1412.6980", "Adam: a method for stochastic optimization"),
    "loshchilov2019": ("10.48550/arXiv.1711.05101", "Decoupled weight decay regularization"),
    "lundberg2017": ("10.48550/arXiv.1705.07874", "A unified approach to interpreting model predictions"),
    "grinsztajn2022": ("10.48550/arXiv.2207.08815", "Why do tree-based models still outperform deep learning on tabular data?"),
    "siddiqi2017": ("10.1002/9781119282396", "Intelligent credit scoring"),
    "hanley1982": ("10.1148/radiology.143.1.7063747", "The meaning and use of the area under a ROC curve"),
    "ferrari2004": ("10.1080/0266476042000214501", "Beta regression for modelling rates and proportions"),
    "yurdakul2020": ("10.21314/jrmv.2020.227", "Statistical properties of the population stability index"),
}
SIN_DOI = {  # se verifican por URL oficial, no por DOI
    "bergstra2011": "https://papers.nips.cc/paper_files/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html",
    "srivastava2014": "https://jmlr.org/papers/v15/srivastava14a.html",
}


def _get(url: str, accept: str) -> str:
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": "Trabajo2-UNAL (verificacion academica)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="replace")


for clave, (doi, esperado) in CANDIDATAS.items():
    print(f"=== {clave} · DOI {doi}\n    esperado: {esperado}")
    try:
        bib = _get(f"https://doi.org/{doi}", "application/x-bibtex").strip()
        print("    BibTeX oficial:", bib)
        meta = json.loads(_get(f"https://api.crossref.org/works/{urllib.request.quote(doi, safe='')}", "application/json"))
        m = meta["message"]
        print("    Crossref issued:", m.get("issued", {}).get("date-parts"), "| published-print:",
              m.get("published-print", {}).get("date-parts"))
    except urllib.error.HTTPError as exc:
        print(f"    ERROR HTTP {exc.code} ({exc.reason})")
    except Exception as exc:  # noqa: BLE001 - se reporta tal cual
        print(f"    ERROR {type(exc).__name__}: {exc}")
for clave, url in SIN_DOI.items():
    print(f"=== {clave} · URL {url}")
    try:
        html = _get(url, "text/html")
        inicio = html.find("<title>")
        print("    título de la página:", html[inicio + 7: html.find("</title>", inicio)].strip()[:200])
    except urllib.error.HTTPError as exc:
        print(f"    ERROR HTTP {exc.code} ({exc.reason})")
    except Exception as exc:  # noqa: BLE001
        print(f"    ERROR {type(exc).__name__}: {exc}")
sys.stdout.flush()
