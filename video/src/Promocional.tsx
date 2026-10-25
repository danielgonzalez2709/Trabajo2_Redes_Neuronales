import { Series, useVideoConfig } from "remotion";
import { Clip } from "./Clip";
import { Cierre } from "./escenas/Cierre";
import { Cifra } from "./escenas/Cifra";
import { Demo } from "./escenas/Demo";
import { DosPreguntas } from "./escenas/DosPreguntas";
import { Equipo } from "./escenas/Equipo";
import { Honesta } from "./escenas/Honesta";
import { RedNeuronal } from "./escenas/RedNeuronal";
import { Titulo } from "./escenas/Titulo";

// Video promocional (Parte E, tarea E6): 2:30 a 30 fps = 4500 fotogramas. Guion en video/guion.md.
// Cada escena es un nodo propio con su duración en línea, para poder ajustarla en el Studio.
// Los clips de los integrantes van en public/clips/ (grabación del sábado 24); mientras falten, se ve un marcador.
export const Promocional: React.FC = () => {
  const { fps } = useVideoConfig();

  return (
    <Series>
      <Series.Sequence name="1 · Título" durationInFrames={300} premountFor={fps}>
        <Titulo texto="¿Me prestas?" />
      </Series.Sequence>
      <Series.Sequence name="2 · El problema (Andrés)" durationInFrames={450} premountFor={fps}>
        <Clip
          archivo=""
          quien="Andrés"
          linea="Cada día millones de personas piden un préstamo. Y del otro lado, un banco tiene que decidir en segundos si confía en ellas."
        />
      </Series.Sequence>
      <Series.Sequence name="3 · Dos preguntas (Camila)" durationInFrames={450} premountFor={fps}>
        <DosPreguntas pregunta1="¿Qué tan probable es que no me paguen?" pregunta2="Y si no me pagan, ¿cuánto pierdo?" />
      </Series.Sequence>
      <Series.Sequence name="4 · Los datos (José)" durationInFrames={450} premountFor={fps}>
        <Cifra />
      </Series.Sequence>
      <Series.Sequence name="5 · Demo de la app (Daniel)" durationInFrames={1200} premountFor={fps}>
        <Demo archivo="" />
      </Series.Sequence>
      <Series.Sequence name="6 · La red (Juan José)" durationInFrames={600} premountFor={fps}>
        <RedNeuronal capas={[5, 6, 4, 1]} texto="Una red neuronal, comparada de forma justa." />
      </Series.Sequence>
      <Series.Sequence name="7 · Honesta (Camila)" durationInFrames={450} premountFor={fps}>
        <Honesta titulo="Y es honesta." subtitulo="Te muestra qué pesa en tu puntaje y qué tan calibradas están sus probabilidades." />
      </Series.Sequence>
      <Series.Sequence name="8 · El equipo" durationInFrames={450} premountFor={fps}>
        <Equipo archivo="" nombres={["Andrés", "Camila", "Daniel", "José", "Juan José"]} />
      </Series.Sequence>
      <Series.Sequence name="9 · Cierre" durationInFrames={150} premountFor={fps}>
        <Cierre llamado="Pruébala." url="URL de la app (pendiente del despliegue)" pie="UNAL · Redes neuronales y algoritmos bioinspirados · 2026-02" />
      </Series.Sequence>
    </Series>
  );
};
