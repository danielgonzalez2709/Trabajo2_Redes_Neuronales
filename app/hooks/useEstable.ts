"use client";
// Adaptador: el valor que deja de cambiar durante `ms` (para anunciar a lectores de pantalla sin repetir cada paso).
import { useEffect, useState } from "react";

export function useEstable<T>(valor: T, ms: number): T {
  const [estable, setEstable] = useState(valor);
  useEffect(() => {
    const t = setTimeout(() => setEstable(valor), ms);
    return () => clearTimeout(t);
  }, [valor, ms]);
  return estable;
}
