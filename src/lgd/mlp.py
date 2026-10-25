"""Perceptrón multicapa de LGD con salida sigmoide (Spec, Parte D §3.3; núcleo §5.3; enunciado Parte 2, punto 3).

Arquitectura: pila ``Linear → activación → Dropout`` por cada entero de ``mlp.capas`` y salida ``Linear(·, 1)``.
La red devuelve el **logit** z; la pérdida (``src/lgd/perdidas.py``, con la derivación) trabaja sobre z en forma
numéricamente estable y ``predict`` aplica la sigmoide, así que la LGD estimada queda en [0, 1].

Bucle de entrenamiento propio mínimo (mientras llega ``entrenar.py`` de la Parte B):
- Semillas: ``fijar_semillas(seed)`` antes de construir la red (pesos iniciales y dropout salen de esa semilla);
  el orden de los mini-lotes sale del ``np.random.Generator`` que devuelve. En CPU se activa
  ``torch.use_deterministic_algorithms(True)``.
- L2: **penalización explícita** ``l2 · Σ‖W‖²`` sobre las matrices de pesos de las capas ``Linear`` (no sobre los
  sesgos), sumada a la pérdida de entrenamiento de cada mini-lote. No se usa ``weight_decay`` del optimizador, para
  que el término sea el mismo con Adam y con SGD.
- Optimizador: ``adam`` (``lr``; ``momento`` no aplica) o ``sgd`` (``lr`` y ``momentum = momento``).
- Parada temprana: tras cada época se calcula la pérdida de validación (modo evaluación, sin dropout y sin el término
  L2); si no mejora durante ``paciencia`` épocas seguidas se detiene y se restauran los pesos de la mejor época.
- Atributos para el reporte por semilla (D4): ``epocas_``, ``mejor_epoca_``, ``device_``, ``tiempo_s_``,
  ``historial_`` (``{"train": [...], "val": [...]}``).
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.common.seeds import fijar_semillas
from src.lgd.perdidas import obtener_perdida

CLAVES_MLP = ("capas", "activacion", "dropout", "l2", "optimizador", "lr", "momento", "batch", "max_epocas",
              "paciencia", "device", "perdida")
ACTIVACIONES = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh}
OPTIMIZADORES = ("adam", "sgd")
DEVICES = ("auto", "cpu", "cuda")


def _leer_cfg_mlp(cfg: dict) -> dict:
    """Bloque ``cfg["mlp"]`` validado; clave faltante → ``KeyError`` y valor inválido → ``ValueError``."""
    if not isinstance(cfg, dict) or "mlp" not in cfg:
        raise KeyError("la configuración no tiene el bloque 'mlp' (núcleo §5.3)")
    mlp = cfg["mlp"]
    faltan = [c for c in CLAVES_MLP if c not in mlp]
    if faltan:
        raise KeyError(f"faltan claves en cfg['mlp']: {faltan}")
    mlp = dict(mlp)
    obtener_perdida(mlp["perdida"])  # valida
    if mlp["activacion"] not in ACTIVACIONES:
        raise ValueError(f"mlp.activacion debe ser una de {list(ACTIVACIONES)}; se recibió {mlp['activacion']!r}")
    if mlp["optimizador"] not in OPTIMIZADORES:
        raise ValueError(f"mlp.optimizador debe ser uno de {list(OPTIMIZADORES)}; se recibió {mlp['optimizador']!r}")
    if mlp["device"] not in DEVICES:
        raise ValueError(f"mlp.device debe ser uno de {list(DEVICES)}; se recibió {mlp['device']!r}")
    if not isinstance(mlp["capas"], (list, tuple)) or not all(
        isinstance(c, int) and not isinstance(c, bool) and c > 0 for c in mlp["capas"]
    ):
        raise ValueError(f"mlp.capas debe ser una lista de enteros positivos; se recibió {mlp['capas']!r}")
    if not 0.0 <= float(mlp["dropout"]) < 1.0:
        raise ValueError(f"mlp.dropout debe estar en [0, 1); se recibió {mlp['dropout']!r}")
    for clave in ("batch", "max_epocas", "paciencia"):
        if not isinstance(mlp[clave], int) or isinstance(mlp[clave], bool) or mlp[clave] < 1:
            raise ValueError(f"mlp.{clave} debe ser un entero >= 1; se recibió {mlp[clave]!r}")
    if float(mlp["lr"]) <= 0 or float(mlp["l2"]) < 0 or float(mlp["momento"]) < 0:
        raise ValueError("mlp.lr debe ser > 0 y mlp.l2, mlp.momento >= 0")
    mlp["capas"] = [int(c) for c in mlp["capas"]]
    return mlp


def _resolver_device(nombre: str) -> torch.device:
    if nombre == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if nombre == "cuda" and not torch.cuda.is_available():
        raise ValueError("mlp.device = 'cuda' pero no hay GPU CUDA disponible; usa 'auto' o 'cpu'")
    return torch.device(nombre)


def _construir_red(n_entradas: int, capas: list[int], activacion: str, dropout: float) -> nn.Sequential:
    modulos: list[nn.Module] = []
    previo = n_entradas
    for ancho in capas:
        modulos += [nn.Linear(previo, ancho), ACTIVACIONES[activacion](), nn.Dropout(dropout)]
        previo = ancho
    modulos.append(nn.Linear(previo, 1))
    return nn.Sequential(*modulos)


def _a_matriz(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X debe ser una matriz 2D; se recibió shape {X.shape}")
    return X


def _a_objetivo(y, n: int, nombre: str) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size != n:
        raise ValueError(f"{nombre} tiene {y.size} valores y X tiene {n} filas")
    if not np.all(np.isfinite(y)) or np.any(y < 0.0) or np.any(y > 1.0):
        raise ValueError(f"{nombre} debe estar en [0, 1] (LGD) y sin faltantes")
    return y


class MLPLGD:
    """Implementa ``ModeloLGD`` (núcleo §5.3)."""

    def __init__(self) -> None:
        self.red_: nn.Sequential | None = None
        self.arquitectura_: dict | None = None
        self.cfg_mlp_: dict | None = None
        self.epocas_: int | None = None
        self.mejor_epoca_: int | None = None
        self.device_: str | None = None
        self.tiempo_s_: float | None = None
        self.historial_: dict | None = None
        self.seed_: int | None = None

    # ------------------------------------------------------------------ entrenamiento
    def fit(self, X_tr, y_tr, X_val, y_val, cfg: dict, seed: int) -> "MLPLGD":
        mlp = _leer_cfg_mlp(cfg)
        perdida = obtener_perdida(mlp["perdida"])
        X_tr = _a_matriz(X_tr)
        X_val = _a_matriz(X_val)
        if X_val.shape[1] != X_tr.shape[1]:
            raise ValueError("X_val y X_tr tienen distinto número de columnas")
        y_tr = _a_objetivo(y_tr, X_tr.shape[0], "y_tr")
        y_val = _a_objetivo(y_val, X_val.shape[0], "y_val")

        inicio = time.perf_counter()
        device = _resolver_device(mlp["device"])
        if device.type == "cpu":
            torch.use_deterministic_algorithms(True)
        rng = fijar_semillas(seed)

        arquitectura = {"n_entradas": int(X_tr.shape[1]), "capas": mlp["capas"], "activacion": mlp["activacion"],
                        "dropout": float(mlp["dropout"])}
        red = _construir_red(**arquitectura).to(device)
        if mlp["optimizador"] == "adam":
            opt = torch.optim.Adam(red.parameters(), lr=float(mlp["lr"]))
        else:
            opt = torch.optim.SGD(red.parameters(), lr=float(mlp["lr"]), momentum=float(mlp["momento"]))
        pesos = [m.weight for m in red.modules() if isinstance(m, nn.Linear)]
        l2 = float(mlp["l2"])

        Xt = torch.from_numpy(X_tr).to(device)
        yt = torch.from_numpy(y_tr).to(device)
        Xv = torch.from_numpy(X_val).to(device)
        yv = torch.from_numpy(y_val).to(device)
        n, batch = Xt.shape[0], mlp["batch"]

        hist_train: list[float] = []
        hist_val: list[float] = []
        mejor_val = float("inf")
        mejor_estado = copy.deepcopy(red.state_dict())
        mejor_epoca = 0
        sin_mejora = 0
        epoca = 0
        for epoca in range(1, mlp["max_epocas"] + 1):
            red.train()
            orden = torch.from_numpy(rng.permutation(n)).to(device)
            suma, cuenta = 0.0, 0
            for i in range(0, n, batch):
                idx = orden[i:i + batch]
                logits = red(Xt[idx]).squeeze(-1)
                datos = perdida(logits, yt[idx])
                total = datos + l2 * sum((w ** 2).sum() for w in pesos) if l2 > 0 else datos
                opt.zero_grad()
                total.backward()
                opt.step()
                suma += float(datos.detach()) * idx.numel()
                cuenta += idx.numel()
            hist_train.append(suma / cuenta)

            red.eval()
            with torch.no_grad():
                val = float(perdida(red(Xv).squeeze(-1), yv))
            hist_val.append(val)
            if val < mejor_val:
                mejor_val, mejor_epoca, sin_mejora = val, epoca, 0
                mejor_estado = copy.deepcopy(red.state_dict())
            else:
                sin_mejora += 1
                if sin_mejora >= mlp["paciencia"]:
                    break

        red.load_state_dict(mejor_estado)
        red.eval()
        self.red_ = red
        self.arquitectura_ = arquitectura
        self.cfg_mlp_ = mlp
        self.epocas_ = epoca
        self.mejor_epoca_ = mejor_epoca
        self.device_ = device.type
        self.tiempo_s_ = time.perf_counter() - inicio
        self.historial_ = {"train": hist_train, "val": hist_val}
        self.seed_ = int(seed)
        return self

    # ------------------------------------------------------------------ inferencia
    def _ajustada(self) -> nn.Sequential:
        if self.red_ is None:
            raise RuntimeError("MLPLGD no está ajustado: llama a fit o cargar primero")
        return self.red_

    def predict(self, X) -> np.ndarray:
        """LGD estimada σ(z) ∈ [0, 1], shape (n,)."""
        red = self._ajustada()
        X = _a_matriz(X)
        if X.shape[1] != self.arquitectura_["n_entradas"]:
            raise ValueError(f"X tiene {X.shape[1]} columnas y la red espera {self.arquitectura_['n_entradas']}")
        device = next(red.parameters()).device
        red.eval()
        with torch.no_grad():
            mu = torch.sigmoid(red(torch.from_numpy(X).to(device)).squeeze(-1))
        return mu.cpu().numpy().astype(np.float64).reshape(-1)

    def n_parametros(self) -> int:
        """Parámetros entrenables de la red."""
        return int(sum(p.numel() for p in self._ajustada().parameters() if p.requires_grad))

    # ------------------------------------------------------------------ persistencia
    def guardar(self, ruta) -> None:
        """Guarda pesos, arquitectura y metadatos: ``cargar`` no necesita la config externa."""
        red = self._ajustada()
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": {k: v.detach().cpu() for k, v in red.state_dict().items()},
                "arquitectura": self.arquitectura_,
                "cfg_mlp": self.cfg_mlp_,
                "epocas": self.epocas_,
                "mejor_epoca": self.mejor_epoca_,
                "device": self.device_,
                "tiempo_s": self.tiempo_s_,
                "historial": self.historial_,
                "seed": self.seed_,
            },
            ruta,
        )

    @classmethod
    def cargar(cls, ruta) -> "MLPLGD":
        """Carga en CPU un modelo guardado con ``guardar``."""
        datos = torch.load(Path(ruta), map_location="cpu", weights_only=True)
        obj = cls()
        obj.red_ = _construir_red(**datos["arquitectura"])
        obj.red_.load_state_dict(datos["state_dict"])
        obj.red_.eval()
        obj.arquitectura_ = datos["arquitectura"]
        obj.cfg_mlp_ = datos["cfg_mlp"]
        obj.epocas_ = datos["epocas"]
        obj.mejor_epoca_ = datos["mejor_epoca"]
        obj.device_ = datos["device"]
        obj.tiempo_s_ = datos["tiempo_s"]
        obj.historial_ = datos["historial"]
        obj.seed_ = datos["seed"]
        return obj
