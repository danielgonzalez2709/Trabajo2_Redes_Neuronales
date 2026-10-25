import { AbsoluteFill, Easing, interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { Clip } from "../Clip";
import { tema } from "../tema";

type Props = { readonly archivo: string };

// Escena 5 (0:55–1:35): la grabación de pantalla de la app dentro de un portátil que entra desde abajo.
export const Demo: React.FC<Props> = ({ archivo }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <AbsoluteFill style={{ backgroundColor: tema.fondo, justifyContent: "center", alignItems: "center" }}>
      <div
        style={{
          width: 1440,
          translate: interpolate(frame, [0, 1.2 * fps], ["0px 200px", "0px 0px"], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.16, 1, 0.3, 1),
          }),
          opacity: interpolate(frame, [0, 0.8 * fps], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }),
        }}
      >
        {/* pantalla */}
        <div
          style={{
            position: "relative",
            width: 1440,
            height: 900,
            borderRadius: 28,
            border: "14px solid #1d1d1f",
            overflow: "hidden",
            backgroundColor: "#111114",
          }}
        >
          <Clip archivo={archivo} quien="Daniel / Camila (grabación de la app)" linea="Llenar el formulario, ver el puntaje y mover el deslizador del ingreso" />
        </div>
        {/* base del portátil */}
        <div style={{ width: 1600, height: 26, margin: "0 -80px", borderRadius: "0 0 24px 24px", backgroundColor: "#2c2c2e" }} />
      </div>
    </AbsoluteFill>
  );
};
