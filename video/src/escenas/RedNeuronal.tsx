import { AbsoluteFill, Easing, interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { tema } from "../tema";

type Props = { readonly capas: number[]; readonly texto: string };

// Escena 6 (1:35–1:55): una red que se ilumina capa por capa y termina en el puntaje.
// `capas` es solo un dibujo (pocas neuronas para que se lea); la arquitectura real la decide la Parte B.
export const RedNeuronal: React.FC<Props> = ({ capas, texto }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const ancho = 1200;
  const alto = 640;
  const x = (c: number) => 120 + (c * (ancho - 240)) / (capas.length - 1);
  const y = (n: number, total: number) => alto / 2 + (n - (total - 1) / 2) * 90;
  const luz = (c: number) =>
    interpolate(frame, [(1 + c * 1.2) * fps, (1.8 + c * 1.2) * fps], [0.15, 1], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
      easing: Easing.bezier(0.16, 1, 0.3, 1),
    });

  return (
    <AbsoluteFill style={{ backgroundColor: tema.fondo, justifyContent: "center", alignItems: "center", gap: 40 }}>
      <svg width={ancho} height={alto}>
        {capas.slice(0, -1).map((n, c) =>
          Array.from({ length: n }).flatMap((_, i) =>
            Array.from({ length: capas[c + 1] }).map((__, j) => (
              <line
                key={`${c}-${i}-${j}`}
                x1={x(c)}
                y1={y(i, n)}
                x2={x(c + 1)}
                y2={y(j, capas[c + 1])}
                stroke={tema.acento}
                strokeWidth={2}
                opacity={0.35 * luz(c + 1)}
              />
            )),
          ),
        )}
        {capas.map((n, c) =>
          Array.from({ length: n }).map((_, i) => (
            <circle key={`n-${c}-${i}`} cx={x(c)} cy={y(i, n)} r={20} fill={c === capas.length - 1 ? tema.acento2 : tema.acento} opacity={luz(c)} />
          )),
        )}
      </svg>
      <div
        style={{
          fontFamily: tema.fuente,
          fontSize: 72,
          fontWeight: 700,
          color: tema.texto,
          opacity: interpolate(frame, [(1.5 + capas.length * 1.2) * fps, (2.3 + capas.length * 1.2) * fps], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        {texto}
      </div>
    </AbsoluteFill>
  );
};
