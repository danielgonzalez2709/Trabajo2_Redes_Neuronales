// Máquina de estados del globo de ayuda (toggletip): hover y foco lo abren de paso; clic o toque lo fija.
// Dominio puro: el componente traduce los eventos del navegador a estos.

export type EstadoToggletip = "cerrado" | "pasajero" | "fijo";

export type EventoToggletip =
  | "entrar" // el ratón entra en el botón o en el globo
  | "salir" // el ratón sale de los dos (tras una pausa breve)
  | "foco" // el botón recibe el foco
  | "desenfoque" // el botón pierde el foco
  | "clic" // clic, Enter/Espacio o toque en el botón
  | "escape"
  | "fuera"; // puntero abajo fuera del botón y del globo

export function transicion(estado: EstadoToggletip, evento: EventoToggletip): EstadoToggletip {
  switch (evento) {
    case "entrar":
    case "foco":
      return estado === "cerrado" ? "pasajero" : estado;
    case "salir":
      return estado === "pasajero" ? "cerrado" : estado;
    case "clic":
      // Abierto de paso (el ratón ya estaba encima o el toque dio el foco): el clic lo fija en vez de cerrarlo.
      return estado === "fijo" ? "cerrado" : "fijo";
    case "desenfoque":
    case "escape":
    case "fuera":
      return "cerrado";
  }
}
