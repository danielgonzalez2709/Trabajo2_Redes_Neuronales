// Tarjeta del resultado: selector de modelo, puntaje y PD del escenario (y su cambio frente al perfil base), los del
// otro modelo y la ubicación en la población.
import type { NombreModelo, Poblacion } from "../lib/contrato";
import type { Tramo } from "../lib/densidad";
import {
  deltaPuntaje, formatearPD, formatearPercentil, formatearPuntaje, formatearPuntos, nombreReferencia,
} from "../lib/formato";
import type { Resultado } from "../lib/modelo";
import { DensidadPoblacion } from "./DensidadPoblacion";
import { COLOR_MODELO, ORDEN_MODELOS, TITULO_MODELO, TITULO_MODELO_FRASE } from "./modelos";
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
  poblacion: Poblacion;
  tramos: Tramo[];
}

export function TarjetaResultado(p: Props) {
  const { modelo, resultados, resultadoBase: rb, poblacion, tramos } = p;
  const r = resultados[modelo];
  const otro: NombreModelo = modelo === "mlp" ? "logistica" : "mlp";
  const ro = resultados[otro];
  const ref = nombreReferencia(poblacion.referencia);
  const pct = formatearPercentil(r.percentil);
  const puntaje = formatearPuntaje(r.score);
  const d = rb ? deltaPuntaje(rb.score, r.score) : 0;
  const tono = d > 0 ? estilos.sube : d < 0 ? estilos.baja : estilos.igual;

  return (
    <section className={estilos.tarjeta} aria-labelledby="titulo-resultado">
      <h2 id="titulo-resultado" className="oculta">
        Tu resultado
      </h2>
      <Segmentado
        nombre="modelo"
        leyenda="Modelo"
        leyendaVisible={false}
        opciones={ORDEN_MODELOS.map((m) => ({ valor: m, etiqueta: TITULO_MODELO[m], color: COLOR_MODELO[m] }))}
        valor={modelo}
        onCambio={(v) => p.onModelo(v as NombreModelo)}
      />

      <div className={estilos.cifras}>
        <div className={estilos.cifra}>
          <span className={estilos.rotulo}>{rb ? "Puntaje del escenario" : "Tu puntaje"}</span>
          <span className={estilos.puntaje}>{puntaje}</span>
        </div>
        <div className={estilos.cifra}>
          <span className={estilos.rotulo}>Probabilidad de incumplimiento</span>
          <span className={estilos.pd}>{formatearPD(r.pd)}</span>
        </div>
      </div>
      <p className="oculta" aria-live="polite" aria-atomic="true">
        {p.anuncio}
      </p>

      {rb ? (
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
      ) : (
        <p className={estilos.nota}>Más puntaje, menos probabilidad de incumplimiento.</p>
      )}

      <div className={estilos.acciones}>
        <button type="button" className={estilos.accion} onClick={p.onGuardarBase} disabled={!rb}>
          Guardar como mi perfil
        </button>
        <button type="button" className={estilos.accion} onClick={p.onVolverABase} disabled={!rb}>
          Volver a mi perfil
        </button>
      </div>

      <p className={estilos.otro}>
        <span className={estilos.punto} style={{ background: COLOR_MODELO[otro] }} aria-hidden="true" />
        <span>
          Con {TITULO_MODELO_FRASE[otro]}: puntaje <strong>{formatearPuntaje(ro.score)}</strong> · PD{" "}
          <strong>{formatearPD(ro.pd)}</strong>
        </span>
      </p>

      <div className={estilos.poblacion}>
        <h3 className={estilos.subtitulo}>Ubicación en la población</h3>
        <p className={estilos.frase}>
          {rb ? "El escenario" : "Tu puntaje"} supera al <strong>{pct} %</strong> de la población de {ref}.
        </p>
        <DensidadPoblacion
          tramos={tramos}
          puntaje={r.score}
          color={COLOR_MODELO[modelo]}
          descripcion={
            `Distribución de los puntajes de la población de ${ref} con ${TITULO_MODELO_FRASE[modelo]}, ` +
            `de ${formatearPuntaje(tramos[0]?.desde ?? r.score)} a ` +
            `${formatearPuntaje(tramos[tramos.length - 1]?.hasta ?? r.score)} puntos. ` +
            `Tu puntaje, ${puntaje}, está marcado y supera al ${pct} % de esa población.`
          }
        />
        <p className={estilos.referencia}>
          Referencia: los puntajes de la población de {ref} ({poblacion.referencia}) con{" "}
          {TITULO_MODELO_FRASE[modelo]}.
        </p>
      </div>
    </section>
  );
}
