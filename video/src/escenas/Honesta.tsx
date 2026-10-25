import { AbsoluteFill, Easing, interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { tema } from "../tema";

type Props = { readonly titulo: string; readonly subtitulo: string };

// Escena 7 (1:55–2:10): "Y es honesta." La curva de calibración real llega de la Parte B (results/data/pd/);
// hasta entonces se muestra solo la diagonal de calibración perfecta, sin inventar puntos.
export const Honesta: React.FC<Props> = ({ titulo, subtitulo }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const lado = 440;
  const trazo = interpolate(frame, [1.5 * fps, 3 * fps], [lado * Math.SQRT2, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

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
      <svg width={lado + 40} height={lado + 40}>
        <rect x={20} y={20} width={lado} height={lado} fill="none" stroke="#3a3a3c" strokeWidth={2} />
        <line
          x1={20}
          y1={lado + 20}
          x2={lado + 20}
          y2={20}
          stroke={tema.verde}
          strokeWidth={6}
          strokeLinecap="round"
          strokeDasharray={lado * Math.SQRT2}
          strokeDashoffset={trazo}
        />
        {/* TODO(dato-faltante): puntos de la curva de calibración del MLP desde results/data/pd/ (Parte B, B7) */}
      </svg>
    </AbsoluteFill>
  );
};
