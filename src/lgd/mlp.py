"""Perceptrón multicapa de LGD con salida sigmoide (Spec, Parte D §3.3; núcleo §5.3; enunciado Parte 2, punto 3).

Arquitectura: pila ``Linear → activación → Dropout`` por cada entero de ``mlp.capas`` y salida ``Linear(·, 1)``.
La red devuelve el **logit** z; la pérdida (``src/lgd/perdidas.py``, con la derivación) trabaja sobre z en forma
numéricamente estable y ``predict`` aplica la sigmoide, así que la LGD estimada queda en [0, 1].

Bucle de entrenamiento propio mínimo (mientras llega ``entrenar.py`` de la Parte B):
- Semillas: ``fijar_semillas(seed)`` antes de construir la red (pesos iniciales y dropout salen de esa semilla);
  el orden de los mini-lotes sale del ``np.random.Generator`` que devuelve. En CPU se activa
  ``torch.use_deterministic_algorithms(True)``.
- L2: penalización L2 explícita en la pérdida, ``l2 · Σ‖W‖²`` sobre las matrices de pesos de las capas ``Linear``,
  sin sesgos; equivale a ``weight_decay = 2·l2`` acoplado sobre los pesos. Con Adam, ni esto ni ``weight_decay``
  son weight decay desacoplado (AdamW).
- Optimizador: ``adam`` (``lr``; ``momento`` no aplica) o ``sgd`` (``lr`` y ``momentum = momento``).
- Parada temprana: tras cada época se calcula la pérdida de validación (modo evaluación, sin dropout y sin el término
  L2); si no mejora durante ``paciencia`` épocas seguidas se detiene y se restauran los pesos de la mejor época.
  Una pérdida de validación no finita detiene el ajuste con ``RuntimeError`` (divergencia).
- Atributos para el reporte por semilla (D4): ``epocas_``, ``mejor_epoca_``, ``device_``, ``tiempo_s_``,
  ``historial_`` (``{"train": [...], "val": [...]}``). ``historial_["train"]`` es el promedio de la pérdida (sin L2)
  de los mini-lotes de la época, medida con dropout activo y con pesos a mitad de época: no es comparable en nivel
  con ``historial_["val"]``.
"""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from src.common.seeds import fijar_semillas, validar_entero
from src.lgd.perdidas import obtener_perdida
from src.lgd.validacion import a_matriz, a_objetivo

CLAVES_MLP = ("capas", "activacion", "dropout", "l2", "optimizador", "lr", "momento", "batch", "max_epocas",
              "paciencia", "device", "perdida")
ACTIVACIONES = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh}
OPTIMIZADORES = ("adam", "sgd")
DEVICES = ("auto", "cpu", "cuda")
# Metadatos del modelo ajustado: atributo ``<nombre>_`` y clave ``<nombre>`` del archivo .pt.
_METADATOS = ("arquitectura", "cfg_mlp", "epocas", "mejor_epoca", "device", "tiempo_s", "historial", "seed")

Perdida = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _leer_cfg_mlp(cfg: dict) -> dict:
    """Bloque ``cfg["mlp"]`` validado (copia); clave faltante → ``KeyError``, valor inválido → ``ValueError`` o
    ``TypeError``. ``mlp.perdida`` la valida ``obtener_perdida``."""
    if not isinstance(cfg, dict) or "mlp" not in cfg:
        raise KeyError("la configuración no tiene el bloque 'mlp' (núcleo §5.3)")
    mlp = dict(cfg["mlp"])
    faltan = [c for c in CLAVES_MLP if c not in mlp]
    if faltan:
        raise KeyError(f"faltan claves en cfg['mlp']: {faltan}")
    if mlp["activacion"] not in ACTIVACIONES:
        raise ValueError(f"mlp.activacion debe ser una de {list(ACTIVACIONES)}; se recibió {mlp['activacion']!r}")
    if mlp["optimizador"] not in OPTIMIZADORES:
        raise ValueError(f"mlp.optimizador debe ser uno de {list(OPTIMIZADORES)}; se recibió {mlp['optimizador']!r}")
    if mlp["device"] not in DEVICES:
        raise ValueError(f"mlp.device debe ser uno de {list(DEVICES)}; se recibió {mlp['device']!r}")
    if not isinstance(mlp["capas"], (list, tuple)):
        raise ValueError(f"mlp.capas debe ser una lista de enteros positivos; se recibió {mlp['capas']!r}")
    mlp["capas"] = [validar_entero(c, f"mlp.capas[{i}]", minimo=1) for i, c in enumerate(mlp["capas"])]
    for clave in ("batch", "max_epocas", "paciencia"):
        mlp[clave] = validar_entero(mlp[clave], f"mlp.{clave}", minimo=1)
    if not 0.0 <= float(mlp["dropout"]) < 1.0:
        raise ValueError(f"mlp.dropout debe estar en [0, 1); se recibió {mlp['dropout']!r}")
    if not float(mlp["lr"]) > 0.0:
        raise ValueError(f"mlp.lr debe ser > 0; se recibió {mlp['lr']!r}")
    if not float(mlp["l2"]) >= 0.0:
        raise ValueError(f"mlp.l2 debe ser >= 0; se recibió {mlp['l2']!r}")
    if not float(mlp["momento"]) >= 0.0:
        raise ValueError(f"mlp.momento debe ser >= 0; se recibió {mlp['momento']!r}")
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


def _crear_optimizador(mlp: dict, parametros) -> torch.optim.Optimizer:
    if mlp["optimizador"] == "adam":
        return torch.optim.Adam(parametros, lr=float(mlp["lr"]))
    return torch.optim.SGD(parametros, lr=float(mlp["lr"]), momentum=float(mlp["momento"]))


def _penalizacion_l2(pesos: list[torch.Tensor]) -> torch.Tensor:
    """Σ‖W‖² sobre las matrices de pesos (sin sesgos)."""
    return sum((w ** 2).sum() for w in pesos)


def _entrenar_una_epoca(red: nn.Sequential, opt: torch.optim.Optimizer, perdida: Perdida, Xt: torch.Tensor,
                        yt: torch.Tensor, orden: torch.Tensor, batch: int, pesos: list[torch.Tensor],
                        l2: float) -> float:
    """Una pasada por los mini-lotes en el orden dado; devuelve la pérdida media (sin L2) ponderada por lote."""
    red.train()
    suma, cuenta = 0.0, 0
    for i in range(0, orden.numel(), batch):
        idx = orden[i:i + batch]
        datos = perdida(red(Xt[idx]).squeeze(-1), yt[idx])
        total = datos + l2 * _penalizacion_l2(pesos)
        opt.zero_grad()
        total.backward()
        opt.step()
        suma += float(datos.detach()) * idx.numel()
        cuenta += idx.numel()
    return suma / cuenta


def _perdida_validacion(red: nn.Sequential, perdida: Perdida, Xv: torch.Tensor, yv: torch.Tensor) -> float:
    """Pérdida de validación en modo evaluación (sin dropout) y sin el término L2."""
    red.eval()
    with torch.no_grad():
        return float(perdida(red(Xv).squeeze(-1), yv))


class MLPLGD:
    """Implementa ``ModeloLGD`` (núcleo §5.3)."""

    def __init__(self) -> None:
        self.red_: nn.Sequential | None = None
        for nombre in _METADATOS:
            setattr(self, f"{nombre}_", None)

    # ------------------------------------------------------------------ entrenamiento
    def fit(self, X_tr, y_tr, X_val, y_val, cfg: dict, seed: int) -> "MLPLGD":
        mlp = _leer_cfg_mlp(cfg)
        perdida = obtener_perdida(mlp["perdida"])
        X_tr = a_matriz(X_tr, nombre="X_tr")
        X_val = a_matriz(X_val, n_columnas=X_tr.shape[1], nombre="X_val")
        y_tr = a_objetivo(y_tr, X_tr.shape[0], nombre="y_tr")
        y_val = a_objetivo(y_val, X_val.shape[0], nombre="y_val")

        inicio = time.perf_counter()
        device = _resolver_device(mlp["device"])
        if device.type == "cpu":
            torch.use_deterministic_algorithms(True)
        rng = fijar_semillas(seed)

        arquitectura = {"n_entradas": int(X_tr.shape[1]), "capas": mlp["capas"], "activacion": mlp["activacion"],
                        "dropout": float(mlp["dropout"])}
        red = _construir_red(**arquitectura).to(device)
        opt = _crear_optimizador(mlp, red.parameters())
        pesos = [m.weight for m in red.modules() if isinstance(m, nn.Linear)]
        Xt, yt = torch.from_numpy(X_tr).to(device), torch.from_numpy(y_tr).to(device)
        Xv, yv = torch.from_numpy(X_val).to(device), torch.from_numpy(y_val).to(device)

        hist_train: list[float] = []
        hist_val: list[float] = []
        mejor_val, mejor_epoca = math.inf, 0
        mejor_estado = copy.deepcopy(red.state_dict())
        for epoca in range(1, mlp["max_epocas"] + 1):
            orden = torch.from_numpy(rng.permutation(Xt.shape[0])).to(device)
            hist_train.append(_entrenar_una_epoca(red, opt, perdida, Xt, yt, orden, mlp["batch"], pesos,
                                                  float(mlp["l2"])))
            val = _perdida_validacion(red, perdida, Xv, yv)
            if not math.isfinite(val):
                raise RuntimeError(f"pérdida de validación no finita ({val}) en la época {epoca}: el entrenamiento "
                                   f"divergió; prueba un mlp.lr menor (actual {mlp['lr']})")
            hist_val.append(val)
            if val < mejor_val:
                mejor_val, mejor_epoca = val, epoca
                mejor_estado = copy.deepcopy(red.state_dict())
            elif epoca - mejor_epoca >= mlp["paciencia"]:
                break

        red.load_state_dict(mejor_estado)
        red.eval()
        self.red_ = red
        self.arquitectura_ = arquitectura
        self.cfg_mlp_ = mlp
        self.epocas_ = len(hist_val)
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
        X = a_matriz(X, n_columnas=self.arquitectura_["n_entradas"])
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
        contenido = {nombre: getattr(self, f"{nombre}_") for nombre in _METADATOS}
        contenido["state_dict"] = {k: v.detach().cpu() for k, v in red.state_dict().items()}
        torch.save(contenido, ruta)

    @classmethod
    def cargar(cls, ruta) -> "MLPLGD":
        """Carga en CPU un modelo guardado con ``guardar``."""
        datos = torch.load(Path(ruta), map_location="cpu", weights_only=True)
        obj = cls()
        for nombre in _METADATOS:
            setattr(obj, f"{nombre}_", datos[nombre])
        obj.red_ = _construir_red(**obj.arquitectura_)
        obj.red_.load_state_dict(datos["state_dict"])
        obj.red_.eval()
        return obj
