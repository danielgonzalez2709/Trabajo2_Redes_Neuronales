// Nombre visible y color por entidad de cada modelo (src/viz/paleta.py: logística #2a78d6, MLP #eb6834).
import type { NombreModelo } from "../lib/contrato";

export const ORDEN_MODELOS: readonly NombreModelo[] = ["mlp", "logistica"];

export const TITULO_MODELO: Record<NombreModelo, string> = {
  mlp: "Red neuronal (MLP)",
  logistica: "Regresión logística",
};

/** Para frases: "Con la regresión logística…". */
export const TITULO_MODELO_FRASE: Record<NombreModelo, string> = {
  mlp: "la red neuronal (MLP)",
  logistica: "la regresión logística",
};

export const COLOR_MODELO: Record<NombreModelo, string> = {
  mlp: "var(--ds-modelo-mlp)",
  logistica: "var(--ds-modelo-logistica)",
};
