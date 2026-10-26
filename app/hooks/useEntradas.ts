"use client";
// Adaptador: estado de las entradas y del modelo elegido, sincronizado con la URL (?modelo=…&loan_amnt=…).
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import type { Entrada, NombreModelo } from "../lib/contrato";
import { escribirParams, leerModelo, leerValores, type Valores } from "../lib/entradas";

const ESPERA_URL_MS = 200; // al arrastrar un deslizador, la URL se escribe al soltar o tras una pausa breve

export interface EstadoEntradas {
  modelo: NombreModelo;
  setModelo: (m: NombreModelo) => void;
  /** null mientras no se conocen las entradas (modelos sin cargar). */
  valores: Valores | null;
  setValor: (nombre: string, v: number | string) => void;
}

export function useEntradas(entradas: Entrada[] | null): EstadoEntradas {
  const params = useSearchParams();
  const router = useRouter();
  const ruta = usePathname();
  const [modelo, setModeloEstado] = useState<NombreModelo>(() => leerModelo(params));
  // Mientras nadie toca nada, los valores salen de la URL; después, del estado (la URL solo lo refleja).
  const [editados, setEditados] = useState<Valores | null>(null);
  const [tocado, setTocado] = useState(false);

  const valores = editados ?? (entradas ? leerValores(entradas, params) : null);

  const setModelo = useCallback((m: NombreModelo) => {
    setModeloEstado(m);
    setTocado(true);
  }, []);

  const setValor = useCallback(
    (nombre: string, v: number | string) => {
      if (!entradas) return;
      setEditados((prev) => ({ ...(prev ?? leerValores(entradas, params)), [nombre]: v }));
      setTocado(true);
    },
    [entradas, params],
  );

  useEffect(() => {
    if (!tocado || !entradas || !valores) return;
    const qs = escribirParams(entradas, modelo, valores).toString();
    if (qs === params.toString()) return;
    const t = setTimeout(() => router.replace(qs ? `${ruta}?${qs}` : ruta, { scroll: false }), ESPERA_URL_MS);
    return () => clearTimeout(t);
  }, [tocado, entradas, valores, modelo, params, router, ruta]);

  return { modelo, setModelo, valores, setValor };
}
