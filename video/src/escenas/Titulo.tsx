import { AbsoluteFill, Easing, interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { tema } from "../tema";

type Props = { readonly texto: string };

// Escena 1 (0:00–0:10): el título aparece letra por letra sobre negro.
export const Titulo: React.FC<Props> = ({ texto }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const letras = Array.from(texto);
  const porLetra = 0.12 * fps;

  return (
    <AbsoluteFill style={{ backgroundColor: tema.fondo, justifyContent: "center", alignItems: "center" }}>
      <div style={{ fontFamily: tema.fuente, fontSize: 200, fontWeight: 700, letterSpacing: "-0.03em", color: tema.texto }}>
        {letras.map((letra, i) => (
          <span
            key={i}
            style={{
              opacity: interpolate(frame, [1 * fps + i * porLetra, 1 * fps + i * porLetra + 0.6 * fps], [0, 1], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
                easing: Easing.bezier(0.16, 1, 0.3, 1),
              }),
              translate: interpolate(
                frame,
                [1 * fps + i * porLetra, 1 * fps + i * porLetra + 0.6 * fps],
                ["0px 40px", "0px 0px"],
                { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.bezier(0.16, 1, 0.3, 1) },
              ),
              display: "inline-block",
              whiteSpace: "pre",
            }}
          >
            {letra}
          </span>
        ))}
      </div>
    </AbsoluteFill>
  );
};
