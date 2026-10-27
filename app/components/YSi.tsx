// "¿Y si…?": los cambios de una sola variable que más suben el puntaje del escenario; tocar uno lo aplica.
import { useRef } from "react";
import type { Entrada } from "../lib/contrato";
import type { Cambio } from "../lib/escenarios";
import { deltaPuntaje, etiquetaSinUnidad, formatearPD, formatearPuntos, formatearValor } from "../lib/formato";
import estilos from "./YSi.module.css";

interface Props {
  entradas: Entrada[];
  cambios: Cambio[];
  /** Frase del modelo elegido: "la red neuronal (MLP)". */
  modeloFrase: string;
  /** El escenario cambió y los cambios aún se recalculan (al arrastrar un deslizador). */
  pendiente: boolean;
  onAplicar: (nombre: string, valor: number | string) => void;
}

export function YSi({ entradas, cambios, modeloFrase, pendiente, onAplicar }: Props) {
  const porNombre = new Map(entradas.map((e) => [e.nombre, e]));
  // La tarjeta tocada desaparece al recalcular: el foco vuelve al encabezado de la sección para no perderse.
  const titulo = useRef<HTMLHeadingElement>(null);
  return (
    <section className={estilos.bloque} aria-labelledby="titulo-ysi" aria-busy={pendiente}>
      <h2 id="titulo-ysi" className={estilos.titulo} ref={titulo} tabIndex={-1}>
        ¿Y si…?
      </h2>
      <p className={estilos.explica}>
        Los cambios de <strong>una</strong> sola variable que más suben el puntaje del escenario según {modeloFrase}.
        Toca uno para probarlo. Son lo que dice el modelo, no consejos.
      </p>
      {cambios.length === 0 ? (
        <p className={estilos.nada}>Ningún cambio de una sola variable sube el puntaje del escenario al menos 1 punto.</p>
      ) : (
        <ol className={`${estilos.lista} ${pendiente ? estilos.pendiente : ""}`}>
          {cambios.map((c) => {
            const e = porNombre.get(c.nombre);
            if (!e) return null;
            return (
              <li key={c.nombre}>
                <button
                  type="button"
                  className={estilos.cambio}
                  disabled={pendiente}
                  onClick={() => {
                    onAplicar(c.nombre, c.hasta);
                    titulo.current?.focus();
                  }}
                >
                  <span className={estilos.que}>
                    {etiquetaSinUnidad(e)}{" "}
                    <span className={estilos.valores}>
                      {formatearValor(e, c.desde)} → {formatearValor(e, c.hasta)}
                    </span>
                  </span>
                  <span className={estilos.efecto}>
                    <strong className={estilos.sube}>{formatearPuntos(deltaPuntaje(c.antes.score, c.despues.score))}</strong>{" "}
                    <span className={estilos.pd}>
                      (PD {formatearPD(c.antes.pd)} → {formatearPD(c.despues.pd)})
                    </span>
                  </span>
                </button>
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}
