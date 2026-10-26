// "Dónde te ubicas": barras de ancho de puntaje igual con la proporción de créditos de la población en cada una
// (lib/densidad); la barra que contiene el puntaje de la persona va en color sólido con su etiqueta encima.
import type { Barra } from "../lib/densidad";
import { indiceBarra } from "../lib/densidad";
import { formatearPercentil, formatearPuntaje } from "../lib/formato";
import estilos from "./Ubicacion.module.css";

const ANCHO = 320;
const ALTO = 150;
const M = { izq: 2, der: 2, arriba: 32, abajo: 20 };
const SEPARACION = 2.5;
const ALTO_MIN = 2.5; // las barras casi vacías se siguen viendo, como una raya
const PILDORA = { ancho: 62, alto: 20 };

interface Props {
  barras: Barra[];
  puntaje: number;
  percentil: number;
  color: string;
  /** "prueba" (nombre de `poblacion.referencia`). */
  referencia: string;
  modeloFrase: string;
}

export function Ubicacion({ barras, puntaje, percentil, color, referencia, modeloFrase }: Props) {
  if (barras.length === 0) return null;
  const min = barras[0].desde;
  const max = barras[barras.length - 1].hasta;
  const masaMax = Math.max(...barras.map((b) => b.masa));
  const base = ALTO - M.abajo;
  const paso = (ANCHO - M.izq - M.der) / barras.length;
  const ancho = paso - SEPARACION;
  const alto = (m: number) => Math.max(ALTO_MIN, (m / masaMax) * (base - M.arriba));
  const tuya = indiceBarra(barras, puntaje);
  const xTuya = M.izq + tuya * paso + SEPARACION / 2 + ancho / 2;
  const yTuya = base - alto(barras[tuya].masa);
  const xPildora = Math.min(Math.max(xTuya, PILDORA.ancho / 2), ANCHO - PILDORA.ancho / 2);
  const p = formatearPuntaje(puntaje);
  const descripcion =
    `Distribución de los puntajes de los créditos de ${referencia} con ${modeloFrase}, de ${formatearPuntaje(min)} a ` +
    `${formatearPuntaje(max)} puntos, en ${barras.length} barras de igual ancho; a la izquierda, más riesgo. Tu puntaje, ` +
    `${p}, está en la barra resaltada, de ${formatearPuntaje(barras[tuya].desde)} a ` +
    `${formatearPuntaje(barras[tuya].hasta)} puntos; tu riesgo estimado es menor que el de aproximadamente el ` +
    `${formatearPercentil(percentil)} % de esos créditos.`;

  return (
    <section className={estilos.tarjeta} aria-labelledby="titulo-ubicacion">
      <header className={estilos.cabeza}>
        <h2 id="titulo-ubicacion" className={estilos.titulo}>
          Dónde te ubicas
        </h2>
        <span className={estilos.referencia}>Créditos de {referencia}</span>
      </header>
      <svg className={estilos.figura} viewBox={`0 0 ${ANCHO} ${ALTO}`} role="img" aria-label={descripcion}>
        {barras.map((b, j) => {
          const h = alto(b.masa);
          return (
            <rect
              key={j}
              className={estilos.barra}
              x={M.izq + j * paso + SEPARACION / 2}
              y={base - h}
              width={ancho}
              height={h}
              rx={Math.min(3, ancho / 2, h / 2)}
              fill={color}
              fillOpacity={j === tuya ? 1 : 0.22}
              stroke={color}
              strokeOpacity={j === tuya ? 1 : 0.7}
              strokeWidth={1}
            />
          );
        })}
        <g className={estilos.marca} style={{ transform: `translate(${xPildora.toFixed(2)}px, ${(yTuya - 8).toFixed(2)}px)` }}>
          <rect
            x={-PILDORA.ancho / 2}
            y={-PILDORA.alto}
            width={PILDORA.ancho}
            height={PILDORA.alto}
            rx={PILDORA.alto / 2}
            fill="var(--ds-tinta)"
          />
          <text x={0} y={-PILDORA.alto / 2} dominantBaseline="central" textAnchor="middle" className={estilos.tu}>
            Tú · {p}
          </text>
        </g>
        <text x={M.izq} y={ALTO - 4} className={estilos.eje}>
          {formatearPuntaje(min)}
        </text>
        <text x={ANCHO - M.der} y={ALTO - 4} textAnchor="end" className={estilos.eje}>
          {formatearPuntaje(max)}
        </text>
      </svg>
      <div className={estilos.sentido} aria-hidden="true">
        <span>← Más riesgo</span>
        <span>Puntaje</span>
        <span>Menos riesgo →</span>
      </div>
      <p className={estilos.pie}>
        Cada barra muestra qué parte de los créditos de {referencia} tiene un puntaje en ese rango (aproximada a partir de
        sus percentiles). La barra de color es la tuya.
      </p>
    </section>
  );
}
