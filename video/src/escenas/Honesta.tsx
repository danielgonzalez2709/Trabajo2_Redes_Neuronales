import { AbsoluteFill, Easing, interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { calibracionPd, calibracionTasa } from "../datos";
import { tema } from "../tema";

type Props = { readonly titulo: string; readonly subtitulo: string };

// Escena 7 (1:55–2:10): "Y es honesta." Curva de calibración real del MLP de PD (semilla mediana, prueba), leída de
// blog/docs/figures/pd-calibracion.json (B7). Los dos ejes comparten la escala [0, L]: los puntos quedan donde caen
// respecto de la diagonal y = x (en prueba, por encima: el modelo subestima la tasa de malos).
// Límite común de los ejes: el máximo de x e y redondeado hacia arriba al múltiplo de 0,1 (el −1e-9 evita el error de coma flotante).
const limite = Math.ceil(Math.max(...calibracionPd, ...calibracionTasa) * 10 - 1e-9) / 10;
// Texto del límite con coma decimal.
const textoLimite = String(limite).replace(".", ",");

export const Honesta: React.FC<Props> = ({ titulo, subtitulo }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const lado = 440;
  // Margen izquierdo e inferior para las etiquetas de los ejes.
  const margen = 70;
  // Margen superior y derecho.
  const borde = 20;
  // Convierte una PD predicha en la coordenada x del SVG.
  const aX = (p: number) => margen + (p / limite) * lado;
  // Convierte una tasa observada en la coordenada y del SVG (el eje y crece hacia arriba).
  const aY = (t: number) => borde + lado - (t / limite) * lado;
  const trazo = interpolate(frame, [1.5 * fps, 3 * fps], [lado * Math.SQRT2, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  // Opacidad de cada punto: aparecen uno a uno desde el segundo 3 (fin de la diagonal), 0,4 s entre puntos.
  const opacidad = (i: number) =>
    interpolate(frame, [(3 + 0.4 * i) * fps, (3.3 + 0.4 * i) * fps], [0, 1], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    });
  // Estilo de las marcas de los ejes.
  const marca = { fontFamily: tema.fuente, fontSize: 22, fill: tema.textoTenue } as const;

  return (
    <AbsoluteFill
      style={{ backgroundColor: tema.fondo, flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 160 }}
    >
      <div style={{ fontFamily: tema.fuente, maxWidth: 900 }}>
        <div style={{ fontSize: 140, fontWeight: 700, letterSpacing: "-0.03em", color: tema.texto, whiteSpace: "nowrap" }}>
          {titulo}
        </div>
        <div style={{ fontSize: 48, color: tema.textoTenue, lineHeight: 1.3, marginTop: 24 }}>{subtitulo}</div>
      </div>
      <svg width={margen + lado + borde} height={borde + lado + margen}>
        <rect x={margen} y={borde} width={lado} height={lado} fill="none" stroke="#3a3a3c" strokeWidth={2} />
        <line
          x1={margen}
          y1={lado + borde}
          x2={lado + margen}
          y2={borde}
          stroke={tema.verde}
          strokeWidth={6}
          strokeLinecap="round"
          strokeDasharray={lado * Math.SQRT2}
          strokeDashoffset={trazo}
        />
        {/* Origen común de los dos ejes. */}
        <text x={margen - 10} y={borde + lado + 28} textAnchor="end" {...marca}>
          0
        </text>
        {/* Extremo del eje x. */}
        <text x={margen + lado} y={borde + lado + 28} textAnchor="end" {...marca}>
          {textoLimite}
        </text>
        {/* Extremo del eje y. */}
        <text x={margen - 10} y={borde + 18} textAnchor="end" {...marca}>
          {textoLimite}
        </text>
        {/* Etiqueta del eje x. */}
        <text x={margen + lado / 2} y={borde + lado + 56} textAnchor="middle" {...marca}>
          PD predicha
        </text>
        {/* Etiqueta del eje y, girada. */}
        <text x={margen - 30} y={borde + lado / 2} textAnchor="middle" transform={`rotate(-90 ${margen - 30} ${borde + lado / 2})`} {...marca}>
          Tasa observada
        </text>
        {/* Tramos que unen cada punto con el anterior; aparecen con el punto de llegada. */}
        {calibracionPd.slice(1).map((p, i) => (
          <line
            key={`tramo-${i}`}
            x1={aX(calibracionPd[i])}
            y1={aY(calibracionTasa[i])}
            x2={aX(p)}
            y2={aY(calibracionTasa[i + 1])}
            stroke={tema.acento2}
            strokeWidth={4}
            opacity={opacidad(i + 1)}
          />
        ))}
        {/* Puntos de la curva de calibración del MLP (x = PD media del bin, y = tasa observada). */}
        {calibracionPd.map((p, i) => (
          <circle key={`punto-${i}`} cx={aX(p)} cy={aY(calibracionTasa[i])} r={9} fill={tema.acento2} opacity={opacidad(i)} />
        ))}
      </svg>
    </AbsoluteFill>
  );
};
