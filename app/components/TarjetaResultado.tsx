// Tarjeta del resultado: selector de modelo; puntaje, PD y percentil del escenario (y su cambio frente al perfil base);
// qué son el puntaje y la PD (con los números del JSON); los del otro modelo y el desempeño del modelo elegido.
import { expresion } from "../lib/avatar";
import type { ModeloPD, NombreModelo } from "../lib/contrato";
import {
  deltaPuntaje, formatearMetricas, formatearPD, formatearPercentil, formatearPuntaje, formatearPuntos, fraseScorecard,
  nombreReferencia,
} from "../lib/formato";
import type { Resultado } from "../lib/modelo";
import { Anillo } from "./Anillo";
import { Avatar } from "./Avatar";
import { COLOR_MODELO, ORDEN_MODELOS, TITULO_MODELO } from "./modelos";
import { Segmentado } from "./Segmentado";
import estilos from "./TarjetaResultado.module.css";

interface Props {
  modelo: NombreModelo;
  onModelo: (m: NombreModelo) => void;
  /** Escenario, con los dos modelos. */
  resultados: Record<NombreModelo, Resultado>;
  /** Perfil base con el modelo elegido; null si el escenario es el perfil base. */
  resultadoBase: Resultado | null;
  onGuardarBase: () => void;
  onVolverABase: () => void;
  /** Texto para lectores de pantalla, ya estabilizado (no cambia en cada paso del deslizador). */
  anuncio: string;
  /** Del JSON del modelo elegido: `scorecard`, `metricas` y `poblacion`. */
  json: Pick<ModeloPD, "scorecard" | "metricas" | "poblacion">;
}

export function TarjetaResultado(p: Props) {
  const { modelo, resultados, resultadoBase: rb, json } = p;
  const r = resultados[modelo];
  const otro: NombreModelo = modelo === "mlp" ? "logistica" : "mlp";
  const ro = resultados[otro];
  const ref = nombreReferencia(json.poblacion.referencia);
  const pct = formatearPercentil(r.percentil);
  const puntaje = formatearPuntaje(r.score);
  const d = rb ? deltaPuntaje(rb.score, r.score) : 0;
  const tono = d > 0 ? estilos.sube : d < 0 ? estilos.baja : estilos.igual;
  const sujeto = rb ? "El riesgo estimado del escenario" : "Tu riesgo estimado";
  const resto = `de los créditos del conjunto de ${ref}.`;

  return (
    <>
      <Segmentado
        nombre="modelo"
        leyenda="Modelo"
        leyendaVisible={false}
        opciones={ORDEN_MODELOS.map((m) => ({ valor: m, etiqueta: TITULO_MODELO[m], color: COLOR_MODELO[m] }))}
        valor={modelo}
        onCambio={(v) => p.onModelo(v as NombreModelo)}
      />

      <section className={estilos.tarjeta} aria-labelledby="titulo-resultado">
        <header className={estilos.cabeza}>
          <Avatar expresion={expresion(r.percentil)} />
          <div className={estilos.titulos}>
            <h2 id="titulo-resultado" className={estilos.titulo}>
              Tu resultado
            </h2>
            <span className={estilos.pildora}>
              <span className={estilos.punto} style={{ background: COLOR_MODELO[modelo] }} aria-hidden="true" />
              {TITULO_MODELO[modelo]}
            </span>
          </div>
        </header>

        <div className={estilos.cifras}>
          <div className={estilos.columna}>
            <div className={estilos.cifra}>
              <span className={estilos.rotulo}>{rb ? "Puntaje del escenario" : "Puntaje"}</span>
              <span className={estilos.puntaje}>{puntaje}</span>
            </div>
            <div className={estilos.cifra}>
              <span className={estilos.rotulo}>Probabilidad de incumplimiento</span>
              <span className={estilos.pd}>{formatearPD(r.pd)}</span>
            </div>
          </div>
          <Anillo
            percentil={r.percentil}
            texto={pct}
            descripcion={`Percentil ${pct}: ${sujeto.toLowerCase()} es menor que el de aproximadamente el ${pct} % ${resto}`}
          />
        </div>
        <p className="oculta" aria-live="polite" aria-atomic="true">
          {p.anuncio}
        </p>

        {rb && (
          <p className={estilos.comparacion}>
            <span>
              Frente a tu perfil: {formatearPuntaje(rb.score)} → {puntaje},{" "}
              <strong className={tono}>
                {d > 0 && <span aria-hidden="true">▲ </span>}
                {d < 0 && <span aria-hidden="true">▼ </span>}
                {formatearPuntos(d)}
              </strong>
            </span>
            <span>
              PD {formatearPD(rb.pd)} → {formatearPD(r.pd)}
            </span>
          </p>
        )}

        <p className={estilos.frase}>
          {sujeto} es menor que el de aproximadamente el <strong>{pct} %</strong> {resto}
        </p>

        <div className={estilos.acciones}>
          <button type="button" className={estilos.accion} onClick={p.onGuardarBase} disabled={!rb}>
            Guardar como mi perfil
          </button>
          <button type="button" className={estilos.accion} onClick={p.onVolverABase} disabled={!rb}>
            Volver a mi perfil
          </button>
        </div>

        <div className={estilos.explicaciones}>
          <div>
            <h3 className={estilos.pregunta}>¿Qué es el puntaje?</h3>
            <p className={estilos.respuesta}>Más alto = menos riesgo. {fraseScorecard(json.scorecard)}</p>
          </div>
          <div>
            <h3 className={estilos.pregunta}>¿Qué es la probabilidad de incumplimiento?</h3>
            <p className={estilos.respuesta}>
              La probabilidad estimada de que el crédito no se pague: que termine declarado incobrable, en impago o con
              atrasos de 31 a 120 días.
            </p>
          </div>
        </div>

        <p className={estilos.otro}>
          <span className={estilos.punto} style={{ background: COLOR_MODELO[otro] }} aria-hidden="true" />
          <span>
            Con {TITULO_MODELO[otro]}: <strong>{formatearPuntaje(ro.score)}</strong> puntos ·{" "}
            <strong>{formatearPD(ro.pd)}</strong> de incumplimiento
          </span>
        </p>

        <p className={estilos.desempeno}>
          Desempeño de este modelo en {ref}: {formatearMetricas(json.metricas)}
        </p>
      </section>
    </>
  );
}
