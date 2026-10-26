// "Tu perfil": los controles de `entradas` en tres bloques, en el orden del JSON.
import type { Entrada } from "../lib/contrato";
import { agrupar, type Valores } from "../lib/entradas";
import { ControlCategorico } from "./ControlCategorico";
import { ControlNumerico } from "./ControlNumerico";
import estilos from "./PanelEntradas.module.css";

interface Props {
  entradas: Entrada[];
  valores: Valores;
  onCambio: (nombre: string, v: number | string) => void;
}

export function PanelEntradas({ entradas, valores, onCambio }: Props) {
  return (
    <section className={estilos.panel} aria-labelledby="titulo-perfil">
      <h2 id="titulo-perfil" className={estilos.titulo}>
        Tu perfil
      </h2>
      <p className={estilos.subtitulo}>Mueve cualquier dato: el resultado se actualiza al instante.</p>
      {agrupar(entradas).map((b) => (
        <fieldset key={b.titulo} className={estilos.bloque}>
          <legend className={estilos.leyenda}>{b.titulo}</legend>
          {b.entradas.map((e) =>
            e.tipo === "num" ? (
              <ControlNumerico
                key={e.nombre}
                entrada={e}
                valor={Number(valores[e.nombre])}
                onCambio={(v) => onCambio(e.nombre, v)}
              />
            ) : (
              <ControlCategorico
                key={e.nombre}
                entrada={e}
                valor={String(valores[e.nombre])}
                onCambio={(v) => onCambio(e.nombre, v)}
              />
            ),
          )}
        </fieldset>
      ))}
    </section>
  );
}
