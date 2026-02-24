# GAUGE CALIBRATION & PHYSICAL ALIGNMENT

## 1. EL PROBLEMA DE LOS "SENSORES CIEGOS" (SENTINEL VALUES)

Durante la auditoria forense del Smoke Test, se detecto que los archivos `GaugesMaxZ_hmax**.csv` y `GaugesVel_V**.csv` contenian un alto porcentaje (hasta el 100% en algunos casos) del valor centinela `-3.40282e+38`.

* **Causa Fisica:** Los sensores (Gauges) en DualSPHysics son puntos fijos en el espacio (X, Y, Z). En el setup heredado de Diego, los sensores estaban ubicados en posiciones estaticas que no se adaptaban a la posicion del bloque.
* **Consecuencia:** Si la ola del tsunami es baja o si el bloque se reposiciona, el sensor queda "en el aire" o detras de la zona de interes, arrojando un error de lectura (el valor float minimo). Esto inutiliza los datos del flujo para el entrenamiento del Machine Learning.

---

## 2. EL ALGORITMO DE REUBICACION DINAMICA (DYNAMIC SENSOR PLACEMENT)

Para el exito de la campana masiva (LHS), el Modulo 1 (`geometry_builder.py`) debe dejar de usar posiciones de sensores hardcodeadas. Se debe implementar la logica de **"Sensores de Persecucion"**.

### Reglas de Posicionamiento Automatico:

Cada vez que el script genere un nuevo XML, debe recalcular las etiquetas `<point>` de los gauges basandose en las coordenadas del bloque (`boulder_pos`):

1. **Gauge de Velocidad de Impacto (V_impact):**
   * *Ubicacion X:* Debe situarse exactamente a una distancia de 2h (donde h es el *smoothing length*) delante de la cara frontal del bloque.
   * *Ubicacion Z:* Debe situarse a una altura de dp x 2 sobre el suelo para capturar la velocidad de la base del flujo, que es la que ejerce el arrastre principal.
2. **Gauge de Altura de Ola (H_impact):**
   * *Ubicacion X:* En la misma coordenada que el sensor de velocidad.
   * *Ubicacion Z:* Se define un rango vertical (desde el suelo hasta Z=1.5m) para que el software integre la superficie libre y entregue la profundidad real del agua al chocar con la roca.

---

## 3. CALCULO DEL SMOOTHING LENGTH (h)

Para colocar los sensores con precision, se debe usar la formula interna de DualSPHysics:

```
h = coefh * sqrt(3 * dp^2)
```

* **Ejemplo:** Para dp = 0.005 y coefh = 1.0, el valor de h es aprox. 0.0086m.
* **Mandato:** Los sensores deben estar lo suficientemente cerca para medir el flujo que golpeara la roca, pero lo suficientemente lejos para no verse afectados por la zona de estancamiento (stagnation point) que genera el propio solido. La distancia recomendada es 2h a 3h.

---

## 4. ALINEACION DE MATERIALES Y FRICCION (CHRONO SETUP)

El "Movimiento Incipiente" es extremadamente sensible a la friccion estatica. Se debe asegurar que la configuracion de materiales en `Floating_Materials.xml` y la vinculacion en el XML principal sean consistentes:

1. **Material del Bloque:** Debe ser siempre `lime-stone` (o el material de roca definido).
2. **Material de la Playa/Rampa:** Debe ser `steel` (acero pulido, segun el laboratorio).
3. **El Par de Contacto:** Chrono calcula la friccion efectiva como el minimo de los coeficientes de los dos materiales en contacto.
   * *Accion:* Se debe verificar que el valor resultante de friccion (mu aprox. 0.15) sea el que el Dr. Moris desea para el experimento de laboratorio, o escalarlo para condiciones reales de campo (mu aprox. 0.5 - 0.7).

---

## 5. VALIDACION DEL ASENTAMIENTO (FTPAUSE)

Para que el Modulo 3 (`data_cleaner.py`) entregue datos limpios, la "Posicion Cero" del bloque no debe ser la del archivo XML, sino la posicion tras el asentamiento.

* **Logica ETL:** El script debe ignorar los datos entre t=0.0s y t=0.5s.
* **Punto de Referencia:** La "Posicion Inicial de Equilibrio" se toma en el instante exacto t = FtPause. Cualquier desplazamiento posterior a ese milisegundo se considera causado exclusivamente por la fuerza hidrodinamica del tsunami.

---

## 6. RESUMEN

Cuando el usuario pida "Generar un nuevo caso", se debe:

1. Leer el STL y calcular su Centro de Masa (CM).
2. Escribir el XML inyectando el CM y la Inercia.
3. **Calcular y escribir las nuevas coordenadas de los Gauges** para que "apunten" al bloque.
4. Asegurar que el `FtPause` sea suficiente para que el bloque no este "volando" al llegar la ola.
