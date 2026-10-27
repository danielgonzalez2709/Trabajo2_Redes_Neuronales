// Tipos del contrato de exportación (núcleo §5.5 del Spec). Dominio puro: sin React ni navegador.

export type NombreModelo = "logistica" | "mlp";
export type Activacion = "relu" | "leaky_relu" | "tanh" | "identidad";

/** Valor crudo de una entrada: número, texto o ausente (null/undefined). */
export type Crudo = number | string | null | undefined;
export type Entradas = Record<string, Crudo>;

export interface EntradaNum {
  nombre: string;
  tipo: "num";
  etiqueta_es: string;
  ayuda_es: string;
  min: number;
  max: number;
  paso: number;
  valor_defecto: number;
}

export interface EntradaCat {
  nombre: string;
  tipo: "cat";
  etiqueta_es: string;
  ayuda_es: string;
  categorias: string[];
  valor_defecto: string;
  etiquetas_categorias: Record<string, string>;
}

export type Entrada = EntradaNum | EntradaCat;

export interface VariableNum {
  nombre: string;
  tipo: "num";
  mediana: number;
  recorte: [number, number];
  log1p: boolean;
  media: number;
  desv: number;
}

export interface VariableCat {
  nombre: string;
  tipo: "cat";
  categorias: string[];
}

export type Variable = VariableNum | VariableCat;

/** `Preprocesador.to_dict()` de Python (núcleo §5.2). */
export interface Preprocesamiento {
  version: number;
  percentiles_recorte: [number, number];
  categoricas: { limpieza: "strip"; vacio: string; no_vista: string };
  orden_numericas: string[];
  columnas_salida: string[];
  variables: Variable[];
}

export interface Capa {
  /** Forma salida x entrada, como torch.nn.Linear.weight. */
  W: number[][];
  b: number[];
  activacion: Activacion;
}

export interface Red {
  capas: Capa[];
  salida: "sigmoid";
}

export interface Scorecard {
  base: number;
  odds_base: number;
  pdo: number;
  factor: number;
  offset: number;
}

export interface Poblacion {
  /** 101 valores p0..p100 del puntaje en test (interpolación lineal). */
  percentiles_score: number[];
  referencia: string;
}

export interface ModeloPD {
  version_contrato: number;
  generado: string;
  commit: string;
  modelo: NombreModelo;
  entradas: Entrada[];
  preprocesamiento: Preprocesamiento;
  red: Red;
  scorecard: Scorecard;
  poblacion: Poblacion;
  metricas: { auc_test: number; ks_test: number; brier_test: number };
}

export interface CasoParidad {
  id: string;
  entradas: Entradas;
  pd: number;
  score: number;
}

export interface CasosParidad {
  version_contrato: number;
  modelo: NombreModelo;
  tolerancia: { pd: number; score: number };
  casos: CasoParidad[];
}
