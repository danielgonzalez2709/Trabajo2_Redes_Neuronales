import { AbsoluteFill, Easing, interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { nModelado, nVariables, periodo } from "../datos";
import { tema } from "../tema";

// Escena 4 (0:40–0:55): el número de créditos cuenta hacia arriba; debajo aparecen tres datos del modelo.
// Las cifras vienen del repositorio (src/datos.ts). No se muestra la tasa de malos por año: su subida reciente es en
// parte un sesgo de selección (blog, sección 2.4) y en un video promocional se leería como un hecho.
export const Cifra: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const entra = (inicio: number) => ({
    opacity: interpolate(frame, [inicio, inicio + 0.8 * fps], [0, 1], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
      easing: Easing.bezier(0.16, 1, 0.3, 1),
    }),
  });
  const datos = [
    { valor: `${periodo.desde}–${periodo.hasta}`, etiqueta: "años de historia" },
    { valor: `${nVariables}`, etiqueta: "datos que tú conoces al pedir" },
    { valor: "0", etiqueta: "datos posteriores al préstamo" },
  ];

  return (
    <AbsoluteFill
      style={{ backgroundColor: tema.fondo, justifyContent: "center", alignItems: "center", fontFamily: tema.fuente, gap: 24 }}
    >
      <div style={{ fontSize: 220, fontWeight: 700, letterSpacing: "-0.04em", color: tema.texto }}>
        {Math.round(
          interpolate(frame, [0.5 * fps, 3.5 * fps], [0, nModelado], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.16, 1, 0.3, 1),
          }),
        ).toLocaleString("es-CO")}
      </div>
      <div style={{ fontSize: 56, color: tema.textoTenue, ...entra(2 * fps) }}>créditos reales de Lending Club</div>
      <div style={{ display: "flex", gap: 140, marginTop: 100 }}>
        {datos.map((d, i) => (
          <div key={d.etiqueta} style={{ textAlign: "center", ...entra((5 + i * 1.2) * fps) }}>
            <div style={{ fontSize: 110, fontWeight: 700, color: tema.acento, letterSpacing: "-0.03em" }}>{d.valor}</div>
            <div style={{ fontSize: 40, color: tema.textoTenue, maxWidth: 420 }}>{d.etiqueta}</div>
          </div>
        ))}
      </div>
    </AbsoluteFill>
  );
};
