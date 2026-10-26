import { describe, expect, it } from "vitest";
import type { Entrada, EntradaNum } from "../lib/contrato";
import {
  deltaPuntaje, etiquetaSinUnidad, formatearDelta, formatearNumero, formatearPD, formatearPercentil, formatearPuntaje,
  formatearPuntos, formatearValor, nombreReferencia, unidad,
} from "../lib/formato";

const num = (etiqueta_es: string): EntradaNum => ({
  nombre: "x", tipo: "num", etiqueta_es, ayuda_es: "", min: 0, max: 1, paso: 1, valor_defecto: 0,
});

describe("formato", () => {
  it("PD en % con 1 decimal y coma decimal", () => {
    expect(formatearPD(0.18349)).toBe("18,3 %");
    expect(formatearPD(0.05)).toBe("5,0 %");
  });

  it("puntaje redondeado a entero solo para mostrar", () => {
    expect(formatearPuntaje(553.49)).toBe("553");
    expect(formatearPuntaje(553.5)).toBe("554");
  });

  it("percentil entero hacia abajo (\"supera al X %\" nunca exagera)", () => {
    expect(formatearPercentil(63.9)).toBe("63");
    expect(formatearPercentil(99.7)).toBe("99");
    expect(formatearPercentil(100)).toBe("100");
    expect(formatearPercentil(0)).toBe("0");
  });

  it("unidad solo si la etiqueta la declara entre paréntesis al final", () => {
    expect(unidad(num("Monto solicitado (USD)"))).toBe("USD");
    expect(unidad(num("Uso de tus cupos rotativos (%)"))).toBe("%");
    expect(unidad(num("Años de historial crediticio"))).toBeNull();
    expect(unidad(num("Líneas de crédito abiertas"))).toBeNull();
  });

  it("etiqueta sin la unidad entre paréntesis", () => {
    expect(etiquetaSinUnidad(num("Monto solicitado (USD)"))).toBe("Monto solicitado");
    expect(etiquetaSinUnidad(num("Plazo"))).toBe("Plazo");
  });

  it("número con separador de miles y los decimales del paso", () => {
    expect(formatearNumero(35000, 25)).toBe("35.000");
    expect(formatearNumero(1350, 25)).toBe("1.350");
    expect(formatearNumero(0.6, 0.1)).toBe("0,6");
    expect(formatearNumero(15, 0.1)).toBe("15,0");
  });

  it("nombre de la población de referencia en español", () => {
    expect(nombreReferencia("test")).toBe("prueba");
    expect(nombreReferencia("otra")).toBe("otra");
  });

  it("valor de una entrada para leer: número con su unidad o etiqueta de la categoría", () => {
    const dti: Entrada = { ...num("Deuda sobre ingreso (%)"), paso: 0.1 };
    expect(formatearValor(dti, 25)).toBe("25,0 %");
    expect(formatearValor({ ...num("Monto solicitado (USD)"), paso: 25 }, 10000)).toBe("10.000 USD");
    expect(formatearValor(num("Líneas de crédito abiertas"), 9)).toBe("9");
    const cat: Entrada = {
      nombre: "term", tipo: "cat", etiqueta_es: "Plazo", ayuda_es: "", categorias: ["36 months", "X"],
      valor_defecto: "36 months", etiquetas_categorias: { "36 months": "36 meses" },
    };
    expect(formatearValor(cat, "36 months")).toBe("36 meses");
    expect(formatearValor(cat, "X")).toBe("X"); // sin etiqueta: el valor tal cual
  });

  it("delta de puntos con signo explícito (signo menos tipográfico)", () => {
    expect(formatearDelta(13)).toBe("+13");
    expect(formatearDelta(-13)).toBe("−13");
    expect(formatearDelta(0)).toBe("0");
  });

  it("delta del puntaje mostrado: diferencia de los enteros que se ven", () => {
    expect(deltaPuntaje(531.6, 545.4)).toBe(13); // 532 -> 545
    expect(deltaPuntaje(545.4, 531.6)).toBe(-13);
    expect(deltaPuntaje(531.6, 531.9)).toBe(0);
  });

  it("puntos con signo y singular", () => {
    expect(formatearPuntos(13)).toBe("+13 puntos");
    expect(formatearPuntos(-1)).toBe("−1 punto");
    expect(formatearPuntos(0)).toBe("0 puntos");
  });
});
