<!-- Responsable: Daniel + todos. Enunciado (Anexo del núcleo), Entregable 7: ≥ 5 hallazgos en ≥ 3 categorías, cada uno con
prompt literal, respuesta, sospecha, evidencia, corrección y lección. Fuente: alucinaciones/registro.md (no copiar a mano
cifras que no estén ahí). Cada parte agrega aquí sus hallazgos cuando estén confirmados. Guía: blog/README.md -->
## 12. Cacería de la alucinación

Registramos cada error de la IA en `alucinaciones/registro.md`, con el prompt literal, la evidencia reproducible (scripts en
`alucinaciones/evidencias/`) y la corrección. La [[tab:caceria]] resume los hallazgos hasta ahora.

::: tabla caceria
Título: Hallazgos de la cacería de la alucinación
Fuente: Elaboración propia a partir de alucinaciones/registro.md y alucinaciones/evidencias/.
| Id | Categoría | Qué dijo o hizo la IA | Qué mostró la evidencia |
|---|---|---|---|
| H-01 | Código | Calculó el índice de estabilidad poblacional con cortes por cuantiles | En variables discretas los cortes colapsan y el índice da 0 aunque la población cambie |
| H-02 | Datos del mundo real | "En `mths_since_last_delinq`, un vacío significa que nunca hubo mora" | El diccionario no lo dice y el 0,46 % de esos vacíos sí tiene moras recientes |
:::
<!-- Partes B, C, D y E: agregar sus hallazgos a la tabla y una subsección por hallazgo. -->

### H-01 — Un índice que da 0 en silencio (Código)

- **Prompt:** el prompt I de la Parte A (sección 11.2), con la especificación que pide el PSI por variable.
- **Respuesta de la IA:** una función que define los intervalos con los cuantiles de entrenamiento.
- **Cómo sospechamos:** el PSI de los registros públicos negativos era exactamente 0,0000 en validación y en prueba, mientras
  las demás variables daban entre 0,003 y 0,27. Un cero exacto en datos reales no es creíble.
- **Evidencia:** tras el recorte, la variable solo toma dos valores; los nueve cuantiles coinciden en un único corte y los
  dos valores caen en el mismo intervalo, así que el índice compara 100 % contra 100 %. Pero la proporción de solicitantes con
  al menos un registro pasa de 4,1 % en entrenamiento a 12,8 % en validación y 19,5 % en prueba
  (`evidencias/H-01_psi_discreto.py`).
- **Corrección:** en variables con pocos valores distintos, cada valor es su propio intervalo; el PSI pasa a 0,107 y 0,269.
  Una prueba con un caso calculado a mano impide que vuelva a pasar.
- **Lección:** probar las métricas también con variables discretas y desconfiar de los ceros exactos.

### H-02 — El significado de un vacío (Datos del mundo real)

- **Prompt:** el mismo prompt I, con la regla de la especificación de citar el significado de las variables del diccionario.
- **Respuesta de la IA:** en el borrador de la sección 2, "variables con más de la mitad de valores vacíos, como
  `mths_since_last_delinq`, donde un vacío significa «nunca ocurrió» y no un dato faltante".
- **Cómo sospechamos:** el subagente revisor contrastó cada afirmación sobre variables con el diccionario y esta no estaba.
- **Evidencia:** el diccionario solo dice "The number of months since the borrower's last delinquency." y ninguna definición
  habla de vacíos. Además, entre los créditos con el campo vacío, el 0,46 % tiene moras en los últimos dos años
  (`evidencias/H-02_mths_since_last_delinq.py`): si el vacío significara "nunca hubo mora", esa proporción sería cero.
- **Corrección:** la sección 2.3 dice ahora que el diccionario no explica el vacío y que por eso no imputamos la variable.
- **Lección:** una explicación "de sentido común" sobre los datos es una afirmación que se cita o se comprueba.

### Verificaciones que salieron correctas

No todo lo que dijo la IA estaba mal, y el enunciado pide documentar también lo verificado. Las 14 referencias que propuso
para el trabajo existen y sus DOI o direcciones oficiales llevan al trabajo esperado (`evidencias/A8_verificacion_bib.*`),
con dos matices de año que dejamos anotados en `blog/references.bib`. Su advertencia de que los créditos recientes ya
resueltos sobrerrepresentan los incumplimientos tempranos se confirmó en los datos (sección 2.4).
