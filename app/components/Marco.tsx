// Piezas fijas de la página: encabezado, pie y avisos de carga o error.
import type { ReactNode } from "react";
import estilos from "./Marco.module.css";

export function Encabezado() {
  return (
    <header className={estilos.encabezado}>
      <h1 className={estilos.titulo}>¿Me prestas?</h1>
      <p className={estilos.lema}>Conoce tu puntaje y tu probabilidad de incumplimiento.</p>
      <p className={estilos.detalle}>
        Describe tu préstamo y tu historial: dos modelos entrenados con préstamos históricos calculan, en tu navegador, qué
        tan probable es que no se pague.
      </p>
    </header>
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
