// Piezas fijas de la página: encabezado (con los enlaces al reporte y al video), pie y avisos de carga o error.
import type { ReactNode } from "react";
import { URL_REPORTE_TECNICO, URL_VIDEO } from "../lib/enlaces";
import estilos from "./Marco.module.css";

export function Encabezado() {
  return (
    <>
      {/* Fuera del <header>: así el título queda fijo arriba mientras se recorre toda la página. */}
      <h1 className={estilos.titulo}>Conoce tu puntaje y tu probabilidad de incumplimiento</h1>
      <header className={estilos.encabezado}>
        <p className={estilos.detalle}>
          Describe tu préstamo y tu historial: dos modelos entrenados con préstamos históricos calculan, en tu
          navegador, qué tan probable es que no se pague.
        </p>
        <nav className={estilos.enlaces} aria-label="Más sobre el proyecto">
          <a className={estilos.enlace} href={URL_REPORTE_TECNICO} target="_blank" rel="noopener noreferrer">
            Reporte técnico<span aria-hidden="true">↗</span>
            <span className="oculta"> (se abre en otra pestaña)</span>
          </a>
          {URL_VIDEO ? (
            <a className={estilos.enlace} href={URL_VIDEO} target="_blank" rel="noopener noreferrer">
              Video promocional<span aria-hidden="true">↗</span>
              <span className="oculta"> (se abre en otra pestaña)</span>
            </a>
          ) : (
            <span className={estilos.enlaceInactivo} aria-disabled="true">
              Video promocional (próximamente)
            </span>
          )}
        </nav>
      </header>
    </>
  );
}

export function Pie() {
  return (
    <footer className={estilos.pie}>
      <p>Herramienta académica; no es una oferta de crédito.</p>
    </footer>
  );
}

export function Aviso({ tipo, children }: { tipo: "carga" | "error"; children: ReactNode }) {
  return (
    <div className={tipo === "error" ? estilos.error : estilos.carga} role={tipo === "error" ? "alert" : "status"}>
      {children}
    </div>
  );
}
