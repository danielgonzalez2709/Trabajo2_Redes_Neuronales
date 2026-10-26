"use client";
// Único contenedor: conecta los adaptadores (modelos, escenario en la URL y perfil base) con los componentes
// presentacionales.
import { Suspense, useDeferredValue, useMemo } from "react";
import { Aviso, Encabezado, Pie } from "../components/Marco";
import { TITULO_MODELO_FRASE } from "../components/modelos";
import { PanelEntradas } from "../components/PanelEntradas";
import { TarjetaResultado } from "../components/TarjetaResultado";
import { YSi } from "../components/YSi";
import { ESPERA_URL_MS, useEntradas } from "../hooks/useEntradas";
import { useEstable } from "../hooks/useEstable";
import { useModelos } from "../hooks/useModelos";
import type { ModeloPD, NombreModelo } from "../lib/contrato";
import { densidadDesdePercentiles } from "../lib/densidad";
import type { Valores } from "../lib/entradas";
import { mejoresCambios, mismosValores, type Cambio } from "../lib/escenarios";
import { deltaPuntaje, formatearPD, formatearPuntaje, formatearPuntos } from "../lib/formato";
import { inferir, type Resultado } from "../lib/modelo";
import estilos from "./page.module.css";

/** Cambios de menos de 1 punto no se proponen: con ≥ 1 el delta mostrado (entero) nunca es "+0". */
const MINIMO_PUNTOS = 1;

interface Calculo {
  resultados: Record<NombreModelo, Resultado>;
  base: Resultado | null;
}

function calcular(
  modelos: Record<NombreModelo, ModeloPD>,
  modelo: NombreModelo,
  valores: Valores,
  base: Valores,
): { calculo: Calculo; error: null } | { calculo: null; error: string } {
  try {
    const resultados = { mlp: inferir(modelos.mlp, valores), logistica: inferir(modelos.logistica, valores) };
    const igual = mismosValores(modelos.mlp.entradas, valores, base);
    return { calculo: { resultados, base: igual ? null : inferir(modelos[modelo], base) }, error: null };
  } catch (e) {
    return { calculo: null, error: e instanceof Error ? e.message : String(e) };
  }
}

function anunciar(c: Calculo | null, modelo: NombreModelo): string {
  if (!c) return "";
  const r = c.resultados[modelo];
  const frente = c.base ? ` (${formatearPuntos(deltaPuntaje(c.base.score, r.score))} frente a tu perfil)` : "";
  return `Puntaje ${formatearPuntaje(r.score)}${frente}; probabilidad de incumplimiento ${formatearPD(r.pd)}.`;
}

function Panel() {
  const carga = useModelos();
  const modelos = carga.estado === "listo" ? carga.modelos : null;
  const { modelo, setModelo, valores, setValor, base, guardarBase, volverABase } = useEntradas(
    modelos?.mlp.entradas ?? null,
  );

  const salida = useMemo(
    () => (modelos && valores && base ? calcular(modelos, modelo, valores, base) : null),
    [modelos, modelo, valores, base],
  );
  const tramos = useMemo(
    () => (modelos ? densidadDesdePercentiles(modelos[modelo].poblacion.percentiles_score) : []),
    [modelos, modelo],
  );

  // El top-3 (unas 450 inferencias) se calcula con el escenario diferido: arrastrar un deslizador no se traba.
  const diferidos = useDeferredValue(valores);
  const cambios = useMemo<Cambio[] | null>(() => {
    if (!modelos || !diferidos) return null;
    try {
      return mejoresCambios(modelos[modelo], diferidos, 3, { minimo: MINIMO_PUNTOS });
    } catch {
      return null; // el error ya se muestra en la tarjeta
    }
  }, [modelos, modelo, diferidos]);

  const anuncio = useEstable(anunciar(salida?.calculo ?? null, modelo), ESPERA_URL_MS);

  if (carga.estado === "error") {
    return <Aviso tipo="error">No se pudieron cargar los modelos: {carga.mensaje}. Recarga la página.</Aviso>;
  }
  if (!modelos || !valores || !base || !salida) return <Aviso tipo="carga">Cargando los modelos…</Aviso>;

  return (
    <div className={estilos.rejilla}>
      <div className={estilos.resultado}>
        {salida.calculo ? (
          <>
            <TarjetaResultado
              modelo={modelo}
              onModelo={setModelo}
              resultados={salida.calculo.resultados}
              resultadoBase={salida.calculo.base}
              onGuardarBase={guardarBase}
              onVolverABase={volverABase}
              anuncio={anuncio}
              poblacion={modelos[modelo].poblacion}
              tramos={tramos}
            />
            {cambios && (
              <YSi
                entradas={modelos.mlp.entradas}
                cambios={cambios}
                modeloFrase={TITULO_MODELO_FRASE[modelo]}
                pendiente={diferidos !== valores}
                onAplicar={setValor}
              />
            )}
          </>
        ) : (
          <Aviso tipo="error">
            No se pudo calcular el resultado: {salida.error}. Revisa los datos del perfil o recarga la página.
          </Aviso>
        )}
      </div>
      <div className={estilos.entradas}>
        <PanelEntradas entradas={modelos.mlp.entradas} valores={valores} base={base} onCambio={setValor} />
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
