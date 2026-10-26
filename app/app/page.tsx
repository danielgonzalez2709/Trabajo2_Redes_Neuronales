"use client";
// Único contenedor: conecta los adaptadores (modelos y entradas en la URL) con los componentes presentacionales.
import { Suspense, useMemo } from "react";
import { Aviso, Encabezado, Pie } from "../components/Marco";
import { PanelEntradas } from "../components/PanelEntradas";
import { TarjetaResultado } from "../components/TarjetaResultado";
import { useEntradas } from "../hooks/useEntradas";
import { useModelos } from "../hooks/useModelos";
import type { ModeloPD, NombreModelo } from "../lib/contrato";
import { densidadDesdePercentiles } from "../lib/densidad";
import type { Valores } from "../lib/entradas";
import { inferir, type Resultado } from "../lib/modelo";
import estilos from "./page.module.css";

function calcular(modelos: Record<NombreModelo, ModeloPD>, valores: Valores) {
  try {
    const resultados: Record<NombreModelo, Resultado> = {
      mlp: inferir(modelos.mlp, valores),
      logistica: inferir(modelos.logistica, valores),
    };
    return { resultados, error: null };
  } catch (e) {
    return { resultados: null, error: e instanceof Error ? e.message : String(e) };
  }
}

function Panel() {
  const carga = useModelos();
  const modelos = carga.estado === "listo" ? carga.modelos : null;
  const { modelo, setModelo, valores, setValor } = useEntradas(modelos?.mlp.entradas ?? null);

  const calculo = useMemo(() => (modelos && valores ? calcular(modelos, valores) : null), [modelos, valores]);
  const tramos = useMemo(
    () => (modelos ? densidadDesdePercentiles(modelos[modelo].poblacion.percentiles_score) : []),
    [modelos, modelo],
  );

  if (carga.estado === "error") {
    return <Aviso tipo="error">No se pudieron cargar los modelos: {carga.mensaje}. Recarga la página.</Aviso>;
  }
  if (!modelos || !valores || !calculo) return <Aviso tipo="carga">Cargando los modelos…</Aviso>;
  if (!calculo.resultados) return <Aviso tipo="error">No se pudo calcular el resultado: {calculo.error}.</Aviso>;

  return (
    <div className={estilos.rejilla}>
      <div className={estilos.resultado}>
        <TarjetaResultado
          modelo={modelo}
          onModelo={setModelo}
          resultados={calculo.resultados}
          poblacion={modelos[modelo].poblacion}
          tramos={tramos}
        />
        {/* E3: aquí va el bloque "¿Y si…?" (R-APP-4), debajo de la tarjeta del resultado. */}
      </div>
      <div className={estilos.entradas}>
        <PanelEntradas entradas={modelos.mlp.entradas} valores={valores} onCambio={setValor} />
      </div>
    </div>
  );
}

export default function Pagina() {
  return (
    <div className={estilos.pagina}>
      <Encabezado />
      <main>
        {/* useSearchParams exige un límite de Suspense en la exportación estática. */}
        <Suspense fallback={<Aviso tipo="carga">Cargando…</Aviso>}>
          <Panel />
        </Suspense>
      </main>
      <Pie />
    </div>
  );
}
