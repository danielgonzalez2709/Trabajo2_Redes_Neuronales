import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import type { Entrada, ModeloPD } from "../lib/contrato";
import { agrupar, escribirParams, leerModelo, leerValores, mismasEntradas, valoresDefecto } from "../lib/entradas";
import { inferir } from "../lib/modelo";

const NUM: Entrada = {
  nombre: "loan_amnt", tipo: "num", etiqueta_es: "Monto (USD)", ayuda_es: "a",
  min: 1000, max: 5000, paso: 25, valor_defecto: 2000,
};
const CAT: Entrada = {
  nombre: "term", tipo: "cat", etiqueta_es: "Plazo", ayuda_es: "b",
  categorias: ["36 months", "60 months"], valor_defecto: "36 months",
  etiquetas_categorias: { "36 months": "36 meses", "60 months": "60 meses" },
};
const ENTRADAS = [NUM, CAT];
const qs = (s: string) => new URLSearchParams(s);

describe("leerValores", () => {
  it("sin parámetros: valores por defecto", () => {
    expect(leerValores(ENTRADAS, qs(""))).toEqual({ loan_amnt: 2000, term: "36 months" });
    expect(valoresDefecto(ENTRADAS)).toEqual({ loan_amnt: 2000, term: "36 months" });
  });

  it("lee valores válidos (números como número)", () => {
    expect(leerValores(ENTRADAS, qs("loan_amnt=3500&term=60+months"))).toEqual({ loan_amnt: 3500, term: "60 months" });
    expect(leerValores(ENTRADAS, qs("loan_amnt=1000"))).toEqual({ loan_amnt: 1000, term: "36 months" }); // bordes
    expect(leerValores(ENTRADAS, qs("loan_amnt=5000")).loan_amnt).toBe(5000);
  });

  it.each(["999", "5001", "abc", "", "  ", "1e9", "NaN", "Infinity", "12abc", "0x10"])(
    "numérica inválida o fuera de rango (%j) -> defecto",
    (v) => {
      expect(leerValores(ENTRADAS, qs(`loan_amnt=${encodeURIComponent(v)}`)).loan_amnt).toBe(2000);
    },
  );

  it.each(["24 months", "36%20MONTHS", "", "OTRA"])("categoría inválida (%j) -> defecto", (v) => {
    expect(leerValores(ENTRADAS, qs(`term=${v}`)).term).toBe("36 months");
  });

  it("ignora parámetros desconocidos", () => {
    expect(leerValores(ENTRADAS, qs("x=1&loan_amnt=3000"))).toEqual({ loan_amnt: 3000, term: "36 months" });
  });
});

describe("leerModelo", () => {
  it("por defecto el MLP; acepta logistica y mlp", () => {
    expect(leerModelo(qs(""))).toBe("mlp");
    expect(leerModelo(qs("modelo=logistica"))).toBe("logistica");
    expect(leerModelo(qs("modelo=mlp"))).toBe("mlp");
    expect(leerModelo(qs("modelo=svm"))).toBe("mlp");
  });
});

describe("escribirParams", () => {
  it("ida y vuelta: escribir y leer devuelve los mismos valores y modelo", () => {
    const valores = { loan_amnt: 4275, term: "60 months" };
    const p = escribirParams(ENTRADAS, "logistica", valores);
    const leido = qs(p.toString());
    expect(leerValores(ENTRADAS, leido)).toEqual(valores);
    expect(leerModelo(leido)).toBe("logistica");
  });

  it("omite lo que está en su valor por defecto (incluido el modelo MLP)", () => {
    expect(escribirParams(ENTRADAS, "mlp", valoresDefecto(ENTRADAS)).toString()).toBe("");
    expect(escribirParams(ENTRADAS, "mlp", { loan_amnt: 3000, term: "36 months" }).toString()).toBe("loan_amnt=3000");
  });

  it("ida y vuelta con los decimales de un paso 0,1", () => {
    const dti: Entrada = { ...NUM, nombre: "dti", min: 0.6, max: 32.5, paso: 0.1, valor_defecto: 15 };
    const p = escribirParams([dti], "mlp", { dti: 0.1 + 0.2 + 12 });
    expect(leerValores([dti], qs(p.toString())).dti).toBe(0.1 + 0.2 + 12);
  });
});

describe("agrupar", () => {
  it("tres bloques en el orden del JSON; lo no listado va al historial", () => {
    const e = (nombre: string): Entrada => ({ ...NUM, nombre });
    const g = agrupar(["loan_amnt", "term", "emp_length", "annual_inc", "purpose", "dti", "pub_rec"].map(e));
    expect(g.map((b) => b.titulo)).toEqual(["El préstamo", "Tú", "Tu historial crediticio"]);
    expect(g.map((b) => b.entradas.map((x) => x.nombre))).toEqual([
      ["loan_amnt", "term", "purpose"],
      ["emp_length", "annual_inc", "dti"],
      ["pub_rec"],
    ]);
  });
});

describe("JSON reales", () => {
  const leer = (n: string) =>
    JSON.parse(readFileSync(join(__dirname, "..", "public", "model", `pd_model_${n}.json`), "utf8")) as ModeloPD;
  const log = leer("logistica");
  const mlp = leer("mlp");

  it("los dos modelos comparten las entradas", () => {
    expect(mismasEntradas(log.entradas, mlp.entradas)).toBe(true);
    expect(mismasEntradas(log.entradas, mlp.entradas.slice(1))).toBe(false);
  });

  it("los 15 controles quedan en los tres bloques, sin perder ninguno", () => {
    const g = agrupar(mlp.entradas);
    expect(g.flatMap((b) => b.entradas)).toHaveLength(mlp.entradas.length);
    expect(g[0].entradas.map((x) => x.nombre)).toEqual(["loan_amnt", "term", "purpose"]);
    expect(g[1].entradas.map((x) => x.nombre)).toEqual(["emp_length", "home_ownership", "annual_inc", "dti"]);
  });

  it("los valores por defecto son válidos (ida y vuelta) y se pueden inferir con los dos modelos", () => {
    const v = valoresDefecto(mlp.entradas);
    expect(leerValores(mlp.entradas, escribirParams(mlp.entradas, "mlp", v))).toEqual(v);
    for (const m of [log, mlp]) {
      const r = inferir(m, v);
      expect(r.pd).toBeGreaterThan(0);
      expect(r.pd).toBeLessThan(1);
    }
  });
});
