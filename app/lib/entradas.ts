// Valores del formulario: por defecto, lectura y escritura en la URL (funciones puras de URLSearchParams) y bloques.
import type { Entrada, NombreModelo } from "./contrato";

export type Valores = Record<string, number | string>;

/** Lo mínimo de URLSearchParams que se usa (también lo cumple el ReadonlyURLSearchParams de Next). */
export interface LectorParams {
  get(nombre: string): string | null;
}

export const MODELO_DEFECTO: NombreModelo = "mlp";
const MODELOS: readonly NombreModelo[] = ["mlp", "logistica"];
const NUMERO = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/; // decimal estricto: sin "0x10", "Infinity" ni vacío

export function valoresDefecto(entradas: Entrada[]): Valores {
  return Object.fromEntries(entradas.map((e) => [e.nombre, e.valor_defecto]));
}

/** Valor válido de una entrada a partir del texto de la URL, o null si falta, no se entiende o está fuera de rango. */
function leerUno(e: Entrada, texto: string | null): number | string | null {
  if (texto === null) return null;
  if (e.tipo === "cat") return e.categorias.includes(texto) ? texto : null;
  const t = texto.trim();
  if (!NUMERO.test(t)) return null;
  const x = Number(t);
  return Number.isFinite(x) && x >= e.min && x <= e.max ? x : null;
}

/** Valores desde la URL; los ausentes, inválidos o fuera de [min, max] toman su `valor_defecto`. */
export function leerValores(entradas: Entrada[], params: LectorParams): Valores {
  return Object.fromEntries(entradas.map((e) => [e.nombre, leerUno(e, params.get(e.nombre)) ?? e.valor_defecto]));
}

export function leerModelo(params: LectorParams): NombreModelo {
  const m = params.get("modelo");
  return MODELOS.find((x) => x === m) ?? MODELO_DEFECTO;
}

/** Parámetros de la URL: solo lo que difiere del valor por defecto (sin nada, la URL queda limpia). */
export function escribirParams(entradas: Entrada[], modelo: NombreModelo, valores: Valores): URLSearchParams {
  const p = new URLSearchParams();
  if (modelo !== MODELO_DEFECTO) p.set("modelo", modelo);
  for (const e of entradas) {
    const v = valores[e.nombre];
    if (v !== undefined && v !== e.valor_defecto) p.set(e.nombre, String(v)); // String(número) se relee exacto
  }
  return p;
}

/** Los dos modelos deben pedir las mismas entradas (mismo formulario para ambos). */
export function mismasEntradas(a: Entrada[], b: Entrada[]): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

export interface Bloque {
  titulo: string;
  entradas: Entrada[];
}

const BLOQUES: { titulo: string; nombres: string[] | null }[] = [
  { titulo: "El préstamo", nombres: ["loan_amnt", "term", "purpose"] },
  { titulo: "Tú", nombres: ["annual_inc", "emp_length", "home_ownership", "dti"] },
  { titulo: "Tu historial crediticio", nombres: null }, // el resto
];

/** Tres bloques; dentro de cada uno, el orden del JSON. */
export function agrupar(entradas: Entrada[]): Bloque[] {
  const listados = new Set(BLOQUES.flatMap((b) => b.nombres ?? []));
  return BLOQUES.map((b) => ({
    titulo: b.titulo,
    entradas: entradas.filter((e) => (b.nombres ? b.nombres.includes(e.nombre) : !listados.has(e.nombre))),
  }));
}
