// Replica de Preprocesador.transform / preprocesar_desde_json (src/common/preprocesamiento.py y
// src/export/exportar_app.py). Función pura, todo en float64.
import type { Crudo, Entradas, Preprocesamiento } from "./contrato";

// Caracteres que str.isspace de Python considera espacio (strip() de Python). El trim() de JavaScript difiere
// (incluye U+FEFF y no incluye \x1c-\x1f), por eso se usa esta clase explícita.
const ESPACIOS = "\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const EXT_INICIO = new RegExp(`^[${ESPACIOS}]+`);
const EXT_FIN = new RegExp(`[${ESPACIOS}]+$`);
const NUMERO = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;

/** Equivalente de `str.strip()` de Python. */
export function stripPy(s: string): string {
  return s.replace(EXT_INICIO, "").replace(EXT_FIN, "");
}

/** Numérica cruda -> número o NaN (vacío). null/undefined/""/solo espacios -> NaN; texto no numérico -> Error. */
function aNumero(x: Crudo, nombre: string): number {
  if (x === null || x === undefined) return NaN;
  if (typeof x === "number") return x; // un NaN de entrada también es vacío
  const t = stripPy(String(x));
  if (t === "") return NaN; // nunca Number(""), que vale 0
  if (!NUMERO.test(t)) throw new Error(`${nombre}: "${x}" no es un número`);
  return Number(t);
}

/** Categórica cruda -> texto limpio: ausente/vacío -> `vacio`; si no, str(x).strip(). */
function limpiarCat(x: Crudo, vacio: string): string {
  if (x === null || x === undefined) return vacio; // nunca String(null)
  if (typeof x === "number" && Number.isNaN(x)) return vacio;
  const t = stripPy(String(x));
  return t === "" ? vacio : t;
}

/** Vector de características (orden de `columnas_salida`) a partir de entradas crudas. */
export function preprocesar(pre: Preprocesamiento, entradas: Entradas): number[] {
  if (pre.version !== 1) throw new Error(`Versión de preprocesador ${pre.version} no soportada`);
  if (pre.categoricas.limpieza !== "strip") throw new Error(`Limpieza desconocida: ${pre.categoricas.limpieza}`);
  const salida: number[] = [];
  for (const v of pre.variables) {
    if (!(v.nombre in entradas)) throw new Error(`Falta la entrada ${v.nombre}`);
    const crudo = entradas[v.nombre];
    if (v.tipo === "num") {
      let x = aNumero(crudo, v.nombre);
      for (const op of pre.orden_numericas) {
        if (op === "imputar_mediana") x = Number.isNaN(x) ? v.mediana : x; // x.fillna(mediana)
        else if (op === "recortar") x = Math.min(Math.max(x, v.recorte[0]), v.recorte[1]); // x.clip(lo, hi)
        else if (op === "log1p_si_aplica") x = v.log1p ? Math.log1p(x) : x; // np.log1p
        else if (op === "estandarizar") x = (x - v.media) / v.desv; // (x - media) / desv
        else throw new Error(`Operación numérica desconocida: ${op}`);
      }
      salida.push(x);
    } else {
      let c = limpiarCat(crudo, pre.categoricas.vacio);
      if (!v.categorias.includes(c)) c = pre.categoricas.no_vista; // cats.where(isin, OTRA)
      for (const cat of v.categorias) salida.push(cat === c ? 1 : 0); // one-hot en el orden de categorias
    }
  }
  if (salida.length !== pre.columnas_salida.length) {
    throw new Error(`Se generaron ${salida.length} columnas y columnas_salida tiene ${pre.columnas_salida.length}`);
  }
  return salida;
}
