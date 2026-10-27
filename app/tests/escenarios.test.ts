import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import type { Entrada, EntradaNum, ModeloPD, VariableNum } from "../lib/contrato";
import { valoresDefecto } from "../lib/entradas";
import { MAX_CANDIDATOS, candidatos, mejoresCambios, mismosValores } from "../lib/escenarios";
import { inferir } from "../lib/modelo";

const num = (nombre: string, min: number, max: number, paso: number, valor_defecto: number): EntradaNum => ({
  nombre, tipo: "num", etiqueta_es: nombre, ayuda_es: "", min, max, paso, valor_defecto,
});

describe("candidatos", () => {
  it("numérica corta: toda la rejilla de min a max con el paso", () => {
    expect(candidatos(num("x", 0, 10, 1, 5))).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  });

  it("paso decimal sin ruido de coma flotante", () => {
    expect(candidatos(num("x", 0.6, 1, 0.1, 0.8))).toEqual([0.6, 0.7, 0.8, 0.9, 1]);
  });

  it("numérica larga: a lo sumo MAX_CANDIDATOS, uniforme, con min y max, todo en el paso", () => {
    const e = num("loan_amnt", 1350, 35000, 25, 10000); // 1 347 puntos
    const c = candidatos(e) as number[];
    expect(c.length).toBeLessThanOrEqual(MAX_CANDIDATOS);
    expect(c.length).toBeGreaterThan(MAX_CANDIDATOS - 5);
    expect(c[0]).toBe(1350);
    expect(c[c.length - 1]).toBe(35000);
    for (let i = 1; i < c.length; i++) expect(c[i]).toBeGreaterThan(c[i - 1]);
    for (const x of c) expect(Number.isInteger((x - 1350) / 25)).toBe(true);
    const saltos = c.slice(1).map((x, i) => x - c[i]);
    expect(Math.max(...saltos) - Math.min(...saltos)).toBeLessThanOrEqual(25); // uniforme salvo el redondeo al paso
  });

  it("categórica: todas sus categorías ofrecidas, en su orden", () => {
    const e: Entrada = {
      nombre: "c", tipo: "cat", etiqueta_es: "c", ayuda_es: "", categorias: ["a", "b"], valor_defecto: "a",
      etiquetas_categorias: {},
    };
    expect(candidatos(e)).toEqual(["a", "b"]);
  });
});

// Logística hecha a mano: z = 0,25·x1 − 0,5·x2 + {a: 0, b: −0,75, c: 0,5} + 0·x3 + 1·x4 − 0,125·x5,
// puntaje = 600 − 10·z (pesos en potencias de 2: los deltas son exactos). x5 se recorta a [0, 2].
const ENTRADAS: Entrada[] = [
  num("x1", 0, 10, 1, 4),
  num("x2", 0, 4, 1, 2),
  { nombre: "c", tipo: "cat", etiqueta_es: "c", ayuda_es: "", categorias: ["a", "b", "c"], valor_defecto: "a",
    etiquetas_categorias: {} },
  num("x3", 0, 3, 1, 1),
  num("x4", 0, 2, 1, 0),
  num("x5", 0, 5, 1, 0),
];
const v = (nombre: string, hi = 1e9): VariableNum =>
  ({ nombre, tipo: "num", mediana: 0, recorte: [-1e9, hi], log1p: false, media: 0, desv: 1 });
const MANO: ModeloPD = {
  version_contrato: 1, generado: "", commit: "", modelo: "logistica",
  entradas: ENTRADAS,
  preprocesamiento: {
    version: 1,
    percentiles_recorte: [1, 99],
    categoricas: { limpieza: "strip", vacio: "DESCONOCIDO", no_vista: "OTRA" },
    orden_numericas: ["imputar_mediana", "recortar", "log1p_si_aplica", "estandarizar"],
    columnas_salida: ["x1", "x2", "c=a", "c=b", "c=c", "x3", "x4", "x5"],
    variables: [
      v("x1"),
      v("x2"),
      { nombre: "c", tipo: "cat", categorias: ["a", "b", "c"] },
      v("x3"),
      v("x4"),
      v("x5", 2),
    ],
  },
  red: { capas: [{ W: [[0.25, -0.5, 0, -0.75, 0.5, 0, 1, -0.125]], b: [0], activacion: "identidad" }], salida: "sigmoid" },
  scorecard: { base: 600, odds_base: 1, pdo: 20, factor: 10, offset: 600 },
  poblacion: { percentiles_score: Array.from({ length: 101 }, (_, k) => 500 + 2 * k), referencia: "test" },
  metricas: { auc_test: 0, ks_test: 0, brier_test: 0 },
};

describe("mejoresCambios con una logística hecha a mano", () => {
  const actual = valoresDefecto(ENTRADAS); // x1 4, x2 2, c a, x3 1, x4 0, x5 0

  it("los 3 mejores: orden por delta, empate por el orden de las entradas, deltas exactos", () => {
    const r = mejoresCambios(MANO, actual);
    expect(r.map((c) => [c.nombre, c.desde, c.hasta, c.delta])).toEqual([
      ["x1", 4, 0, 10], // Δz = −1
      ["x2", 2, 4, 10], // Δz = −1 (empata con x1: va después por el orden de entradas)
      ["c", "a", "b", 7.5],
    ]);
  });

  it("cada cambio trae el resultado de partida y el nuevo, iguales a inferir", () => {
    const [primero] = mejoresCambios(MANO, actual);
    expect(primero.antes).toEqual(inferir(MANO, actual));
    expect(primero.despues).toEqual(inferir(MANO, { ...actual, x1: 0 }));
    expect(primero.despues.score - primero.antes.score).toBe(primero.delta);
  });

  it("sin delta > 0 no aparece (x3 sin efecto, x4 ya en su mejor valor); empate dentro de una variable -> el más cercano", () => {
    const r = mejoresCambios(MANO, actual, 10);
    expect(r.map((c) => c.nombre)).toEqual(["x1", "x2", "c", "x5"]);
    expect(r[3]).toMatchObject({ desde: 0, hasta: 2, delta: 2.5 }); // 2, 3, 4 y 5 dan lo mismo (recorte en 2)
  });

  it("k limita y cada variable aparece a lo sumo una vez", () => {
    expect(mejoresCambios(MANO, actual, 2).map((c) => c.nombre)).toEqual(["x1", "x2"]);
    const r = mejoresCambios(MANO, actual, 10);
    expect(new Set(r.map((c) => c.nombre)).size).toBe(r.length);
  });

  it("si nada sube el puntaje, lista vacía", () => {
    const optimo = { x1: 0, x2: 4, c: "b", x3: 1, x4: 0, x5: 2 };
    expect(mejoresCambios(MANO, optimo)).toEqual([]);
  });

  it("minimo: solo los cambios que suben más que el mínimo", () => {
    expect(mejoresCambios(MANO, actual, 10, { minimo: 7.5 }).map((c) => c.nombre)).toEqual(["x1", "x2"]);
  });
});

describe("mismosValores", () => {
  it("compara solo las entradas del contrato", () => {
    const a = valoresDefecto(ENTRADAS);
    expect(mismosValores(ENTRADAS, a, { ...a })).toBe(true);
    expect(mismosValores(ENTRADAS, a, { ...a, x1: 5 })).toBe(false);
    expect(mismosValores(ENTRADAS, a, { ...a, otra: 1 })).toBe(true);
  });
});

describe.each(["logistica", "mlp"] as const)("mejoresCambios con el JSON real (%s)", (nombre) => {
  const m = JSON.parse(
    readFileSync(join(__dirname, "..", "public", "model", `pd_model_${nombre}.json`), "utf8"),
  ) as ModeloPD;
  const base = valoresDefecto(m.entradas);

  it("cambios válidos, ordenados, sin repetir variable; al aplicarlos el puntaje sube el delta", () => {
    const r = mejoresCambios(m, base);
    expect(r.length).toBeGreaterThan(0);
    expect(r.length).toBeLessThanOrEqual(3);
    expect(new Set(r.map((c) => c.nombre)).size).toBe(r.length);
    for (let i = 1; i < r.length; i++) expect(r[i].delta).toBeLessThanOrEqual(r[i - 1].delta);
    for (const c of r) {
      expect(c.delta).toBeGreaterThan(0);
      expect(c.hasta).not.toBe(c.desde);
      expect(inferir(m, { ...base, [c.nombre]: c.hasta }).score - inferir(m, base).score).toBeCloseTo(c.delta, 9);
    }
  });

  it("costo (solo se reporta el tiempo; no se compara con un umbral)", () => {
    mejoresCambios(m, base); // calentamiento
    const n = 20;
    const t0 = performance.now();
    for (let i = 0; i < n; i++) mejoresCambios(m, base);
    const ms = (performance.now() - t0) / n;
    const evaluaciones = m.entradas.reduce((s, e) => s + candidatos(e).length, 0);
    console.log(`mejoresCambios (${nombre}): ${ms.toFixed(2)} ms por llamada, ${evaluaciones} candidatos`);
    expect(evaluaciones).toBeGreaterThan(0);
  });
});
