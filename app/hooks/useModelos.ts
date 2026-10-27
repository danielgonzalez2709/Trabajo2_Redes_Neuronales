"use client";
// Adaptador: carga los dos `pd_model_*.json` una sola vez (los casos de paridad nunca se cargan en el cliente).
import { useEffect, useState } from "react";
import type { ModeloPD, NombreModelo } from "../lib/contrato";
import { mismasEntradas } from "../lib/entradas";

export type EstadoModelos =
  | { estado: "cargando" }
  | { estado: "error"; mensaje: string }
  | { estado: "listo"; modelos: Record<NombreModelo, ModeloPD> };

async function cargar(nombre: NombreModelo, signal: AbortSignal): Promise<ModeloPD> {
  const r = await fetch(`/model/pd_model_${nombre}.json`, { signal });
  if (!r.ok) throw new Error(`no se pudo descargar el modelo ${nombre} (HTTP ${r.status})`);
  const m = (await r.json()) as ModeloPD;
  if (m.modelo !== nombre) throw new Error(`el archivo de ${nombre} declara el modelo ${m.modelo}`);
  return m;
}

export function useModelos(): EstadoModelos {
  const [estado, setEstado] = useState<EstadoModelos>({ estado: "cargando" });

  useEffect(() => {
    const control = new AbortController();
    Promise.all([cargar("mlp", control.signal), cargar("logistica", control.signal)])
      .then(([mlp, logistica]) => {
        if (!mismasEntradas(mlp.entradas, logistica.entradas)) {
          throw new Error("los dos modelos no piden las mismas entradas");
        }
        setEstado({ estado: "listo", modelos: { mlp, logistica } });
      })
      .catch((e: unknown) => {
        if (control.signal.aborted) return; // desmontado: no se actualiza el estado
        setEstado({ estado: "error", mensaje: e instanceof Error ? e.message : String(e) });
      });
    return () => control.abort();
  }, []);

  return estado;
}
