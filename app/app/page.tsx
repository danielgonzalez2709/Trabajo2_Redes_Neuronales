"use client";

import { useEffect, useState } from "react";
import type { Entradas, ModeloPD, NombreModelo } from "../lib/contrato";
import { inferir, type Resultado } from "../lib/modelo";

const NOMBRES: NombreModelo[] = ["mlp", "logistica"];
const TITULO: Record<NombreModelo, string> = { mlp: "Red neuronal (MLP)", logistica: "Regresión logística" };

function valoresDefecto(m: ModeloPD): Entradas {
  return Object.fromEntries(m.entradas.map((e) => [e.nombre, e.valor_defecto]));
}

// Página mínima de E1: la UI completa es E2.
export default function Pagina() {
  const [resultados, setResultados] = useState<Partial<Record<NombreModelo, Resultado>>>({});
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all(
      NOMBRES.map(async (n) => {
        const r = await fetch(`/model/pd_model_${n}.json`);
        if (!r.ok) throw new Error(`No se pudo cargar el modelo ${n} (${r.status})`);
        const m = (await r.json()) as ModeloPD;
        return [n, inferir(m, valoresDefecto(m))] as const;
      }),
    )
      .then((pares) => setResultados(Object.fromEntries(pares)))
      .catch((e: Error) => setError(e.message));
  }, []);

  return (
    <main style={{ fontFamily: "system-ui, sans-serif", padding: "2rem", maxWidth: 640, margin: "0 auto" }}>
      <h1>¿Me prestas?</h1>
      <p>Resultado con los valores por defecto del formulario.</p>
      {error && <p role="alert">{error}</p>}
      {NOMBRES.map((n) => (
        <section key={n}>
          <h2>{TITULO[n]}</h2>
          {resultados[n] ? (
            <p>
              Puntaje {resultados[n].score.toFixed(1)} · PD {(resultados[n].pd * 100).toFixed(1)} % · percentil{" "}
              {resultados[n].percentil.toFixed(0)}
            </p>
          ) : (
            <p>Calculando…</p>
          )}
        </section>
      ))}
    </main>
  );
}
