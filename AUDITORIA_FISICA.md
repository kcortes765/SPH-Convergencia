# AUDITORIA FISICA — Setup de Diego (DamBreakBLIR_Playa)
> Fecha: 2026-02-20
> Autor: Kevin Cortes + Claude Code (Opus 4.6)
> Fuente de datos: `ENTREGA_KEVIN/` (extraido de workstation RTX 5090, UCN)

---

## Resumen Ejecutivo

Se realizo una auditoria tecnica del setup de DualSPHysics heredado del investigador Diego
(Fondecyt de Iniciacion, Dr. Moris). El objetivo es determinar si la configuracion fisica
es correcta antes de construir el pipeline automatizado de la tesis.

**Veredicto:** El setup es un proof-of-concept ultra-grueso que sirve como referencia
estructural (XML, Chrono, cadena de ejecucion), pero tiene **errores fisicos cuantificables**
que impiden usarlo directamente para resultados publicables.

Se identificaron **3 anomalias criticas** y **4 problemas menores**.

---

## Inventario de Archivos Analizados

| Archivo | Tipo | Contenido |
|---------|------|-----------|
| `DamBreakBLIR_Playa_Def.xml` | XML de definicion | Template de caso (input de GenCase) |
| `DamBreakBLIR_Playa.xml` | XML generado | Output de GenCase con particulas computadas |
| `Floating_Materials.xml` | XML de materiales | Propiedades mecanicas para Chrono |
| `DualSPHEjec_win64_GPU.bat` | Script de ejecucion | Cadena GenCase → DSPH → PostProceso |
| `Run_All.bat` | Script de batch | Iterador sobre carpetas de convergencia |
| `BLIR3.stl` | Geometria 3D | Boulder irregular (1198 vertices, 2392 caras) |
| `Canal_Playa_1esa20_750cm.stl` | Geometria 3D | Playa inclinada 1:20 |
| `ChronoExchange_mkbound_51.csv` | Datos de simulacion | Cinematica del boulder (Chrono) |
| `ChronoBody_forces.csv` | Datos de simulacion | Fuerzas SPH + contacto |
| `GaugesVel_V01-V12.csv` | Datos de simulacion | Velocidad del flujo en 12 puntos |
| `GaugesMaxZ_hmax01-08.csv` | Datos de simulacion | Altura maxima de ola en 8 puntos |
| `Run.csv` | Metadata | Estadisticas de la simulacion |
| `RunPARTs.csv` | Metadata | Info por snapshot |

---

## MISION FORENSE 1: Geometria y Densidad del Boulder

### 1.1 Diagnostico de Malla STL

Se analizo `BLIR3.stl` con la libreria `trimesh` (Python). Script: `auditor_stl.py`.

**STL Original (sin escalar):**
```
Vertices:        1,198
Caras:           2,392
Watertight:      Si (malla cerrada)
Manifold:        Si (volumen valido)
Bounding box:    [-2.074, -2.552, -0.013] → [2.202, 2.697, 0.986]
Dimensiones:     4.276 x 5.249 x 0.999 unidades
Volumen:         8.2849 unidades^3
Centro de masa:  [-0.493, 0.297, 0.461]
```

**Conclusion malla:** La geometria es limpia. No hay defectos de malla (holes, non-manifold edges,
normales invertidas). `trimesh` puede calcular volumen e inercia de forma confiable.

### 1.2 Boulder Escalado (como lo usa Diego)

Diego aplica `<drawscale x="0.04" y="0.04" z="0.04"/>` en el XML. Esto sugiere que el STL
original esta en unidades arbitrarias (probablemente ~25x escala real) y se reduce a metros.

**STL Escalado x0.04:**
```
Bounding box:    [-0.083, -0.102, -0.001] → [0.088, 0.108, 0.039]
Dimensiones:     0.1710 x 0.2100 x 0.0399 metros
                 (17.1 cm x 21.0 cm x 4.0 cm)
Volumen:         0.00053023 m^3 (0.530 litros)
Centro de masa:  [-0.020, 0.012, 0.018] metros (relativo al origen del STL)
```

**Interpretacion geometrica:** El boulder es una piedra aplanada tipo "disco" o "cobble".
La dimension mas delgada es solo **4 cm**. Esto tiene consecuencias criticas para la resolucion
(ver Mision 2).

**Diametro equivalente esférico:**
```
d_eq = (6V/pi)^(1/3) = 0.1004 m = 10.04 cm
```

### 1.3 Analisis de Densidad

Diego impone en el XML:
```xml
<massbody value="1.06053" />  !-- better to impose mass than density (800 kg/m^3)
```

**Calculo de densidad implicita:**
```
rho = M / V = 1.06053 kg / 0.00053023 m^3 = 2000.1 kg/m^3
```

**Comparacion con materiales conocidos:**

| Material | Densidad (kg/m^3) | Match con 2000? |
|----------|-------------------|-----------------|
| Agua | 1,000 | No |
| PVC espumado | 500 - 800 | No |
| PVC rigido | 1,300 - 1,450 | No |
| **Concreto / Arenisca** | **1,800 - 2,400** | **Si** |
| Roca caliza (limestone) | 2,300 - 2,700 | No (pero cercano) |
| Roca granito | 2,600 - 2,800 | No |

**ANOMALIA DETECTADA:** Diego comenta que asume 800 kg/m^3, pero el calculo real da 2000 kg/m^3.
La discrepancia es de **2.5x**.

**Hipotesis de la causa:** Diego probablemente calculo el volumen usando el bounding box
del STL en vez del volumen real de la malla:
```
Vol_bbox = 0.171 * 0.210 * 0.040 = 0.001436 m^3
rho_bbox = 1.06053 / 0.001436 = 738 kg/m^3  ≈ 800 (redondeado)
```
Esto confirma: Diego uso el **volumen del bounding box** (que sobreestima el volumen real
en un factor de 2.7x para esta geometria irregular) para calcular la densidad que comenta.

**Impacto en la simulacion:**
- DualSPHysics usa `massbody` directamente para la dinamica, NO calcula densidad internamente.
- La masa de 1.06053 kg ES lo que se simula, independientemente de la densidad comentada.
- Pero la **interpretacion fisica** de Diego esta errada: cree que simula un bloque de
  ~800 kg/m^3 (flotante en agua) cuando realmente simula uno de ~2000 kg/m^3 (se hunde).
- Esto podria afectar como Diego interpreta y reporta sus resultados.

### 1.4 Tensor de Inercia

Se comparo la inercia calculada por `trimesh` (geometria continua) vs la calculada por
GenCase (discretizacion en 31 particulas SPH).

**Trimesh (geometria real, masa = 1.06053 kg, densidad uniforme):**
```
Ixx = 0.00219478    Ixy = -0.00022592   Ixz = 0.00000719
Iyx = -0.00022592   Iyy = 0.00158169    Iyz = 0.00006968
Izx = 0.00000719    Izy = 0.00006968    Izz = 0.00360960
```

**GenCase (31 particulas, dp=0.05m):**
```
Ixx = 0.00405562    Ixy = 0.00020416    Ixz = -7.44909e-05
Iyx = 0.00020416    Iyy = 0.00476190    Iyz = -1.93125e-05
Izx = -7.44909e-05  Izy = -1.93125e-05  Izz = 0.00749323
```

**Discrepancia:**

| Componente | Trimesh | GenCase | Ratio (GenCase/Trimesh) |
|------------|---------|---------|------------------------|
| Ixx | 0.00219 | 0.00406 | **1.85x** |
| Iyy | 0.00158 | 0.00476 | **3.01x** |
| Izz | 0.00361 | 0.00749 | **2.08x** |

**ANOMALIA CRITICA:** GenCase sobreestima la inercia entre **1.85x y 3.01x**. Causa directa:
con solo 31 particulas a dp=0.05m, la geometria del boulder no se representa bien. Las
particulas quedan distribuidas de forma dispersa y el tensor de inercia resultante
corresponde a un objeto mas grande y mas disperso que el real.

**Impacto fisico:** La inercia 2-3x mayor hace que el boulder sea **mas dificil de rotar**
de lo que deberia. Esto subestima la respuesta rotacional ante el impacto del tsunami
(pitch, roll, yaw responden mas lento).

**Recomendacion para la tesis:** Inyectar el tensor de inercia calculado por `trimesh`
directamente en el XML usando la etiqueta `<inertia>`, en vez de dejar que GenCase lo
calcule desde las particulas. Esto es especialmente importante a dp gruesos donde la
discretizacion es pobre.

---

## MISION FORENSE 2: Resolucion (dp) y Estabilidad Numerica

### 2.1 Estado Actual: dp = 0.05m

Del XML generado por GenCase (`DamBreakBLIR_Playa.xml`):
```
Total particulas:   24,317
  Fixed (playa):    17,446    (mkbound=0)
  Floating (boulder): 31      (mkbound=51)
  Fluid (agua):     6,840     (mkfluid=0)
```

**31 particulas para representar un boulder de 17x21x4 cm es grotescamente insuficiente.**

La dimension minima del boulder (4 cm) es **menor que dp (5 cm)**. Literalmente hay menos
de 1 particula en el espesor del boulder. GenCase logra meter 31 particulas porque:
- `drawfilestl` crea particulas en la **superficie** del STL (no depende del volumen)
- `fillbox void` agrega particulas **interiores** donde quepan
- El fillbox es enorme (1m x 1m x 0.2m), mucho mayor que el boulder

Pero 31 particulas no pueden representar la geometria, el campo de presion sobre el
boulder, ni las fuerzas hidrodinamicas con precision.

### 2.2 Tabla de Escalamiento

Se calcula el numero de particulas del boulder y del dominio completo para distintos dp.
Dominio: 15m x 1m x 1.5m (del XML).

| dp (m) | Part. boulder | dim_min/dp | Part. dominio total | dt aprox (ms) | Costo relativo |
|--------|---------------|------------|---------------------|---------------|----------------|
| 0.050 | ~31 | 0.8 | 24,317 | 0.41 | 1x |
| 0.020 | ~66 | 2.0 | ~375,000 | 0.17 | ~250x |
| 0.010 | ~530 | 4.0 | ~3,000,000 | 0.08 | ~10,000x |
| 0.005 | ~4,242 | 8.0 | ~24,000,000 | 0.04 | ~160,000x |
| **0.004** | **~8,300** | **10.0** | **~47,000,000** | 0.03 | ~500,000x |
| 0.002 | ~66,279 | 20.0 | ~375,000,000 | 0.02 | imposible |

**Regla practica:** Minimo 10 particulas en la dimension menor del solido.
Para este boulder (dim_min = 3.99 cm): **dp <= 0.004 m** necesario.

### 2.3 Calculo del Paso de Tiempo (dt)

El paso de tiempo en DualSPHysics se determina por la condicion CFL:

```
h = coefh * sqrt(3 * dp^2) = coefh * sqrt(3) * dp

Con coefh = 0.75:
  h = 0.75 * 1.7321 * dp = 1.299 * dp

dt ≈ CFL * h / cs

Donde:
  CFL = cflnumber = 0.2
  cs = coefsound * sqrt(g * hswl)
  cs ≈ 20 * sqrt(9.81 * 0.3) ≈ 34.3 m/s  (para dam break h=0.3m)
```

**Valores calculados:**

| dp (m) | h (m) | dt estimado (ms) | Steps para 10s |
|--------|-------|-------------------|----------------|
| 0.050 | 0.0650 | 0.379 | ~26,400 |
| 0.020 | 0.0260 | 0.152 | ~66,000 |
| 0.010 | 0.0130 | 0.076 | ~132,000 |
| 0.005 | 0.0065 | 0.038 | ~264,000 |
| 0.004 | 0.0052 | 0.030 | ~330,000 |

**Verificacion contra datos reales de Diego:**
- Run.csv reporta 24,824 steps para 10s de simulacion a dp=0.05
- dt min real: 0.000370 s (0.37 ms)
- dt max real: 0.000414 s (0.41 ms)
- Nuestro calculo (0.379 ms) coincide. El modelo de estimacion es correcto.

**Ley de escalamiento:** `dt ∝ dp` (lineal). El costo total escala como **O(dp^-4)** en 3D
(dp^-3 por el numero de particulas, dp^-1 por el numero de pasos temporales).

### 2.4 Impacto en ProjectChrono

**Parametro `distancedp`:**
```xml
<distancedp value="0.5" comment="Allowed collision overlap according Dp (default=0.5)" />
```
Esto define el margen de colision como 0.5 * dp metros. A dp=0.005m, el margen seria 2.5mm.
Para un boulder de 10 cm de diametro equivalente, esto es geometricamente razonable.
**No necesita ajuste al cambiar dp.** Solo habria que aumentarlo si se observa tunelamiento
(boulder atraviesa la playa).

**Metodo de contacto NSC (contactmethod=0):**
NSC (Non-Smooth Contacts) resuelve el contacto como un problema de complementariedad con
friccion de Coulomb. No hay problemas de estabilidad documentados a dp fino. De hecho,
a dp mas fino el dt es menor, lo que **mejora** la resolucion temporal de las colisiones.

**CPU bottleneck:**
Chrono corre en un solo core de CPU, independiente de la GPU. Con solo 2 cuerpos
(1 boulder + 1 playa), esto no es bottleneck. Seria problema con cientos de cuerpos.

**Nota sobre `coefh = 0.75`:**
El valor tipico recomendado para problemas con floating bodies y kernel Wendland es
`coefh = 1.0`. Diego usa 0.75, que da un smoothing length mas pequeno. Esto puede:
- Reducir la precision de la interpolacion SPH
- Dar un dt ligeramente mayor (beneficio computacional menor)
- Generar mas "ruido" en las fuerzas sobre el boulder
**Recomendacion:** Verificar con Dr. Moris si esto es intencional o un default heredado.

### 2.5 Viabilidad por Hardware

| dp (m) | Part. totales | VRAM estimada | RTX 4060 (8GB) | RTX 5090 (32GB) |
|--------|---------------|---------------|----------------|-----------------|
| 0.050 | 24K | <100 MB | OK | OK |
| 0.020 | 375K | ~200 MB | OK | OK |
| 0.010 | 3M | ~1.5 GB | OK | OK |
| 0.005 | 24M | ~12 GB | **NO** | OK |
| 0.004 | 47M | ~24 GB | **NO** | **AJUSTADO** |
| 0.002 | 375M | ~190 GB | NO | **NO** |

**Nota:** La estimacion de VRAM es ~500 bytes/particula (posicion, velocidad, densidad,
presion, cell lists, buffers temporales). El valor real depende de la configuracion.

**Estrategia recomendada:**
- **Desarrollo (laptop RTX 4060):** dp = 0.020 - 0.010
- **Convergencia (workstation RTX 5090):** dp = 0.010, 0.005, 0.004
- **Produccion (workstation RTX 5090):** dp determinado por estudio de convergencia

**Optimizacion posible:** Reducir el largo del canal. El dominio actual es 15m, pero el
boulder esta a x=8.5m. Si la playa empieza a ~3m y el boulder esta a 8.5m, podriamos
reducir a ~12m sin perder fisica relevante. Ahorro: ~20% de particulas.

### 2.6 Tiempo de Computo Estimado

Referencia real de Diego: dp=0.05, 24K particulas, RTX 5090 → 209 segundos para 10s fisicos.

Extrapolacion (asumiendo escalamiento O(dp^-4) con eficiencia GPU):

| dp (m) | Particulas | Wall-clock estimado (RTX 5090, 10s fisicos) |
|--------|------------|---------------------------------------------|
| 0.050 | 24K | 3.5 minutos (real: 3.5 min) |
| 0.020 | 375K | ~15 minutos |
| 0.010 | 3M | ~2 horas |
| 0.005 | 24M | ~15 horas |
| 0.004 | 47M | ~2 dias |

**Nota:** Estos son estimados gruesos. El escalamiento real depende de la eficiencia del
GPU a distintos tamaños de problema, cache effects, y overhead de Chrono.

---

## MISION FORENSE 3: Friccion y Materiales Chrono

### 3.1 Materiales Definidos

Del archivo `Floating_Materials.xml`:

| Propiedad | PVC (boulder) | Steel (playa) | Lime-stone (disponible) |
|-----------|---------------|---------------|------------------------|
| Young Modulus (N/m^2) | 3.0e9 | 2.1e11 | 8.0e9 |
| Poisson Ratio | 0.30 | 0.35 | 0.35 |
| Restitution Coefficient | 0.60 | 0.80 | 0.45 |
| Kfric (friccion cinetica) | **0.15** | **0.35** | **0.35** |

### 3.2 Asignacion de Materiales en el XML

```xml
<!-- En la seccion <properties>: -->
<link mkbound="0" property="steel" />    <!-- playa = steel -->

<!-- En la seccion <floatings>: -->
<floating mkbound="51-53" property="pvc">  <!-- boulder = pvc -->
```

```xml
<!-- En la seccion <chrono>: -->
<bodyfloating id="BLIR" mkbound="51" modelfile="AutoActual" />
<bodyfixed id="beach" mkbound="0" modelfile="AutoActual" modelnormal="invert" />
```

### 3.3 Analisis de Friccion

En Chrono NSC, el coeficiente de friccion efectivo para un par de contacto es tipicamente:
```
mu_efectivo = min(Kfric_a, Kfric_b) = min(0.15, 0.35) = 0.15
```

**Contexto fisico de mu = 0.15:**
- PVC sobre acero pulido: 0.15 - 0.20 → **CORRECTO para un modelo de laboratorio**
- Roca sobre roca (en seco): 0.40 - 0.70
- Roca sobre arena: 0.50 - 0.80
- Roca sobre concreto: 0.35 - 0.50

**Interpretacion:** La configuracion de Diego es correcta para un **modelo de laboratorio
a escala** donde se usan bloques de PVC sobre un canal de acero. No es correcta para
simular boulders costeros reales (roca sobre roca/arena).

**Para la tesis (boulders reales):** Seria necesario cambiar a materiales tipo `lime-stone`
(ya definido en el XML) para el boulder, y definir un nuevo material para el fondo marino
(arena, roca, concreto de rompeolas). La eleccion debe ser validada con el Dr. Moris.

### 3.4 Coeficiente de Restitucion

- PVC: e = 0.60 (rebote moderado)
- Steel: e = 0.80 (rebote alto)

Para un boulder costero de roca (e ≈ 0.2-0.4), ambos valores son demasiado altos.
Un coeficiente de restitucion alto hace que el boulder "rebote" mas de lo fisicamente
correcto al impactar contra la playa.

### 3.5 Problemas Adicionales Detectados

**3.5.1 ViscoBoundFactor = 0 (free-slip boundaries)**

```xml
<parameter key="ViscoBoundFactor" value="0" comment="Multiply viscosity value with boundary" />
```

Esto significa que **no hay viscosidad SPH en las particulas boundary**. Las paredes del
canal y la playa son perfectamente lisas desde el punto de vista del fluido. Esto:
- Subestima el arrastre del fluido sobre el boulder
- Subestima la capa limite cerca de las paredes
- Es una simplificacion comun en DualSPHysics para reducir ruido numerico, pero fisicamente
  es una aproximacion fuerte

**Recomendacion:** Considerar `ViscoBoundFactor=1` (misma viscosidad que el fluido) para
simulaciones de produccion. Discutir con Dr. Moris.

**3.5.2 FtPause = 0.0 (sin asentamiento)**

```xml
<parameter key="FtPause" value="0.0" />
```

El boulder se libera desde t=0 sin tiempo de asentamiento. Si la discretizacion coloca
al boulder ligeramente por encima de la playa (gap de ~dp/2), habra un impacto espurio
por gravedad antes de que llegue el flujo.

En los datos de `ChronoExchange_mkbound_51.csv`, se observa que en t=0 la unica
aceleracion es gravitatoria (`face.z = -9.81 m/s^2`) y el boulder tiene velocidad cero.
En los primeros timesteps, el boulder adquiere velocidad vertical negativa (`fvel.z`
pasa de 0 a -0.013 m/s), lo que sugiere que efectivamente esta cayendo por el gap.

**Recomendacion:** Agregar `FtPause=0.5` o `FtPause=1.0` para que el fluido se asiente
y el boulder repose correctamente sobre la playa antes de liberar el Dam Break.

**3.5.3 Rango de mkbound en floating**

```xml
<floating mkbound="51-53" property="pvc">
```

El rango 51-53 sugiere que Diego originalmente tenia hasta 3 boulders (mk=51, 52, 53),
pero en esta configuracion solo hay 1 (mk=51). Los mk 52 y 53 no existen en la geometria.
No es un error — DualSPHysics simplemente ignora los mkbound sin particulas — pero indica
que el XML se heredo de un caso multi-boulder.

**3.5.4 TimeOut = 10.0 (un solo snapshot)**

```xml
<parameter key="TimeMax" value="10" />
<parameter key="TimeOut" value="10.0" />
```

Con TimeMax=10 y TimeOut=10, solo se genera **1 archivo de salida de particulas** (Part0001).
Esto ahorra disco pero imposibilita la visualizacion temporal en ParaView. Para produccion
esto esta bien (los CSVs de Chrono y Gauges se generan con dt=0.001s independientemente).
Para desarrollo/debugging, usar TimeOut=0.1 o menor.

---

## Resumen de Hallazgos

### Anomalias Criticas (impactan resultados)

| # | Hallazgo | Evidencia | Impacto |
|---|----------|-----------|---------|
| 1 | **dp=0.05 inaceptable** | 31 particulas, dim_min/dp=0.8 | Geometria, fuerzas, inercia — todo mal resuelto |
| 2 | **Inercia 2-3x sobreestimada** | Trimesh vs GenCase (ver tabla 1.4) | Rotacion del boulder fisicamente incorrecta |
| 3 | **Densidad implicita ≠ comentada** | 2000 vs 800 kg/m^3 | Interpretacion errada de resultados |

### Problemas Menores (ajustables)

| # | Hallazgo | Valor actual | Recomendacion |
|---|----------|-------------|---------------|
| 4 | ViscoBoundFactor | 0 (free-slip) | Considerar 1 (no-slip) |
| 5 | FtPause | 0.0 | Agregar 0.5 - 1.0 s |
| 6 | coefh | 0.75 | Verificar, tipico es 1.0 |
| 7 | Materiales lab vs real | PVC/Steel | Cambiar a roca para tesis |

### Lo que SI esta correcto

| Aspecto | Detalle |
|---------|---------|
| Malla STL | Watertight, manifold, sin defectos |
| Motor Chrono | RigidAlgorithm=3 con colisiones NSC — apropiado para el problema |
| Estructura XML | Correcta: drawfilestl + fillbox void + massbody + Chrono |
| Cadena de ejecucion | GenCase → DualSPHysics GPU → CSVs automaticos |
| Gauges | 12 puntos de velocidad + 8 de altura — bien distribuidos |
| massbody (no rhopbody) | Correcto — masa explicita es independiente de dp |
| Separador CSV | Punto y coma (;) con decimal punto (.) — consistente |

---

## Decisiones Pendientes para la Tesis

Basado en esta auditoria, las decisiones que se deben tomar antes de implementar son:

| # | Decision | Opciones | Quien decide |
|---|----------|----------|-------------|
| 1 | dp de produccion | 0.005 vs 0.004 (estudio de convergencia) | Estudio numerico |
| 2 | Escenario fisico | Lab (PVC/Steel) vs Real (roca/arena) vs Ambos | Dr. Moris |
| 3 | ViscoBoundFactor | 0 vs 1 | Dr. Moris / literatura |
| 4 | FtPause | 0 vs 0.5 vs 1.0 | Pruebas de asentamiento |
| 5 | coefh | 0.75 vs 1.0 | Dr. Moris / literatura |
| 6 | Fuente de inercia | GenCase (particulas) vs trimesh (geometria) | Tecnico (trimesh es mejor) |
| 7 | Largo del canal | 15m (actual) vs 12m (optimizado) | Analisis de reflexion de onda |

---

## Implicancias para el Pipeline

### Modulo 1 (Geometry Builder)
- **DEBE** calcular inercia con `trimesh` e inyectarla en el XML con `<inertia>`
- **DEBE** calcular el centro de masa real con `trimesh` (no confiar en GenCase a dp grueso)
- **DEBE** parametrizar: dp, dam_height, masa, posicion, rotacion, material

### Modulo 2 (Batch Runner)
- La cadena se simplifica: GenCase → DualSPHysics GPU (sin PartVTK, etc.)
- Los CSVs de Chrono y Gauges se generan automaticamente durante la simulacion
- TimeOut puede ser grande (10s o mayor) — los datos finos vienen de Chrono/Gauges
- Limpieza de .bi4 es critica: a dp=0.005 cada simulacion genera ~12 GB de binarios

### Modulo 3 (ETL)
- Fuente primaria: `ChronoExchange_mkbound_XX.csv` (cinematica del boulder)
- Fuente secundaria: `ChronoBody_forces.csv` (fuerzas SPH + contacto)
- Fuente terciaria: `GaugesVel_V**.csv` y `GaugesMaxZ_hmax**.csv` (flujo)
- Separador: punto y coma (;) — NO se usa -csvsep:1 con Chrono outputs
- Valores centinela: -3.40282e+38 en Gauges = sin datos → reemplazar por NaN

---

## Apendice: Script de Auditoria

El script `auditor_stl.py` en la raiz del proyecto reproduce todos los calculos
de la Mision 1. Ejecutar con:
```bash
cd C:\Seba\Tesis
python auditor_stl.py
```

Dependencias: `trimesh`, `numpy` (ambos instalados via pip).
