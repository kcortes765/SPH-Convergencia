# Educacion Tecnica — SPH-IncipientMotion

> Este documento explica el proyecto completo: la fisica, el software, los parametros y como se conecta todo. Pensado para que lo releas cada vez que un termino tecnico no te suene.

---

## PARTE 1: LA FISICA (El "por que" de todo)

### 1.1 El Fenomeno Real

Imagina la costa de Bahia Cisnes, Atacama. Hay rocas enormes (algunas de 40 toneladas) esparcidas en una planicie detras de la playa. Esas rocas no estaban ahi originalmente — estaban en un acantilado frente al mar. Hace ~600 anos, un tsunami las arranco de su lugar y las arrastro tierra adentro.

**La pregunta de tu tesis:** ¿Que tan grande tuvo que ser ese tsunami para mover una roca de 40 toneladas? ¿Y una de 10? ¿Y una de 1? ¿Importa si la roca es redonda o alargada?

Eso es el **movimiento incipiente**: el momento exacto en que el flujo rompe el equilibrio de la roca y esta empieza a moverse. Un milisegundo antes, la roca esta quieta. Un milisegundo despues, esta siendo arrastrada.

### 1.2 Las Fuerzas en Juego

En el instante critico ($t_{crit}$), hay una pelea entre dos equipos de fuerzas:

```
EQUIPO "MUEVETE" (fuerzas del agua)       EQUIPO "QUEDATE" (fuerzas de la roca)
─────────────────────────────              ──────────────────────────────────────
Fuerza de Arrastre (Drag)                  Peso (gravedad × masa)
  → el agua empuja la roca horizontalmente   → la roca se resiste a subir

Fuerza de Sustentacion (Lift)              Friccion (suelo × roca)
  → el agua intenta levantar la roca         → el suelo agarra la roca

Momento (torque)                           Inercia rotacional
  → el agua intenta voltear la roca          → la roca se resiste a girar
```

Cuando el equipo "muevete" supera al equipo "quedate" → **falla**. La roca se mueve.

Las formulas empiricas clasicas (Nandasena, Engel & May) intentan predecir esto, pero tratan la roca como un **prisma rectangular perfecto**. En la realidad, las rocas costeras son irregulares, asiametricas, y su forma cambia dramaticamente como interactuan con el flujo. Ahi es donde entra tu tesis.

### 1.3 ¿Por que no se puede resolver con una ecuacion?

Porque la interaccion fluido-estructura (FSI) en un tsunami es **caotica**:
- El agua se rompe, salpica, genera vortices
- La superficie libre se deforma violentamente
- La roca puede rotar, deslizarse, volcarse, o combinaciones
- La geometria irregular de la roca crea patrones de presion unicos
- Todo pasa en milisegundos

No hay ecuacion analitica que capture todo esto. Por eso se usa **simulacion numerica**.

---

## PARTE 2: EL METODO SPH (Como simulamos agua)

### 2.1 La Idea Central

SPH = **Smoothed Particle Hydrodynamics** (Hidrodinamica de Particulas Suavizadas).

En lugar de dividir el espacio en cubitos fijos (como hace una malla tradicional de elementos finitos), SPH representa el fluido como **millones de particulas** que se mueven libremente.

Imagina un vaso de agua, pero en lugar de agua continua, tienes millones de bolitas microscopicas. Cada bolita sabe:
- Donde esta (posicion x, y, z)
- A que velocidad se mueve (velocidad vx, vy, vz)
- Que presion siente (presion)
- Que tan densa esta su zona (densidad)

Las bolitas se influencian mutuamente: si una se mueve, empuja a sus vecinas. Si muchas se juntan, la presion sube y se repelen. Esto emerge naturalmente de las ecuaciones de Navier-Stokes discretizadas en particulas.

### 2.2 ¿Por que SPH y no Malla?

Los metodos de malla (como OpenFOAM, ANSYS Fluent) dividen el espacio en cubitos fijos. Funcionan bien para flujos tranquilos, pero cuando el agua se rompe violentamente (como en un tsunami), la malla se deforma y el calculo explota.

SPH no tiene malla. Las particulas simplemente van a donde la fisica las lleva. Por eso es ideal para:
- Olas rompiendo
- Salpicaduras
- Impactos violentos contra solidos
- Cuerpos flotantes que se mueven libremente

**DualSPHysics** es el software open-source que implementa SPH y lo corre en GPU (NVIDIA CUDA). "Dual" porque puede correr en CPU o GPU.

### 2.3 Los Parametros de SPH (los que tienes que defender en tu tesis)

#### `dp` — Distancia entre Particulas (Inter-Particle Distance)

**La variable mas importante de todo el proyecto.**

`dp` define que tan separadas estan las particulas inicialmente. Es la "resolucion" de tu simulacion. Igual que los pixeles en una foto:

```
dp = 0.02m (grueso)          dp = 0.005m (fino)
  ○   ○   ○   ○                ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○
  ○   ○   ○   ○                ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○
  ○   ○   ○   ○                ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○
  16 particulas                ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○
  (borroso, rapido)            256 particulas (nitido, lento)
```

Relacion cubica: `N_particulas ≈ Volumen / dp³`. Reducir dp a la mitad → **8x mas particulas** → **~10-16x mas tiempo**.

#### `h` — Smoothing Length (Longitud de Suavizado)

Cada particula "ve" a sus vecinas dentro de un radio. Ese radio es `h`:

```
        ╭─────────╮
       ╱           ╲
      │    ○  ○     │
      │  ○  ●  ○    │  ← La particula ● siente a todas las ○ dentro del circulo
      │    ○  ○     │     El radio del circulo es h (o 2h segun el kernel)
       ╲           ╱
        ╰─────────╯
```

`h` se calcula automaticamente desde dp: **h = coefh × sqrt(3 × dp²)** en 3D (con coefh = 1.0 por defecto). No lo configuras directamente, pero debes entenderlo porque:
- Si h es muy chico, las particulas "no se ven" entre si → simulacion inestable
- Si h es muy grande, todo se suaviza demasiado → pierdes detalle
- Los evaluadores de tu tesis te van a preguntar por h

#### `CFL Number` — Condicion de Courant-Friedrichs-Lewy

Controla el tamano del paso temporal (`dt`). Es una condicion de estabilidad:

```
dt ≤ CFL × h / velocidad_maxima
```

Traduccion: "el paso de tiempo debe ser lo suficientemente pequeno para que ninguna particula viaje mas de una fraccion de h en un solo paso". Si dt es muy grande, las particulas "saltan" sobre sus vecinas y la simulacion explota numericamente.

- CFL tipico: **0.2** (conservador, estable)
- DualSPHysics ajusta dt automaticamente en cada paso
- Tu solo defines el CFL maximo en el XML; el software calcula dt solo

#### Viscosidad Artificial (`Visco`)

El agua real tiene viscosidad (resistencia al flujo). SPH necesita un modelo de viscosidad para que las particulas no se comporten de forma caotica. Hay dos opciones:

- **Artificial** (`ViscoTreatment=1`): Un truco numerico simple. Valor tipico: `Visco=0.1`. Es lo mas usado en DualSPHysics para flujos con superficie libre.
- **Laminar+SPS** (`ViscoTreatment=2`): Modelo fisico mas realista (incluye turbulencia Sub-Particle Scale). Valor tipico: `Visco=1e-6`. Mas caro computacionalmente.

Para tu tesis, probablemente uses artificial con 0.1 (es el estandar en la literatura de DualSPHysics para dam breaks).

#### `DensityDT` — Delta-SPH (Difusion de Densidad)

Problema: en SPH puro, la densidad de las particulas oscila con ruido numerico (como estática en una radio). Delta-SPH suaviza esas oscilaciones.

- `DensityDT=0`: Sin correccion (ruidoso)
- `DensityDT=1`: Molteni (clasico)
- `DensityDT=2`: **Fourtakas** (recomendado en v5.2+, corrige problemas en la superficie libre)

#### Kernel (Funcion Nucleo)

Define la "forma" de la influencia entre particulas. Piensalo como: ¿como decae la influencia de una particula sobre otra a medida que se alejan?

- **Cubic Spline** (`Kernel=1`): Clasico, rapido
- **Wendland** (`Kernel=2`): Mas suave, mejor para cuerpos flotantes. **Recomendado para tu caso.**

---

## PARTE 3: EL DOMINIO COMPUTACIONAL (El "escenario" virtual)

### 3.1 El Tanque Virtual

Tu simulacion ocurre dentro de una caja 3D. Esta caja es el **dominio computacional**. En la realidad seria un canal de laboratorio, pero en tu caso es 100% digital.

```
     z (arriba)
     │
     │  ┌─────────────────────────────────────┐ pointmax (xmax, ymax, zmax)
     │  │                                     │
     │  │           AIRE (vacio)              │
     │  │                                     │
     │  │   ████                              │
     │  │   ████ ← Columna de agua           │        ⬡ ← Boulder
     │  │   ████    (dam break)              │       ⬡⬡⬡
     │  │   ████                              │      ⬡⬡⬡⬡
     │  │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│ ← Suelo
     │  └─────────────────────────────────────┘
     │  pointmin (xmin, ymin, zmin)
     └──────────────────────────────────── x (largo)
```

### 3.2 `pointmin` y `pointmax`

Son las **esquinas opuestas** del dominio. Definen los limites del "universo" de tu simulacion. Toda particula que salga de estos limites es eliminada.

```xml
<definition dp="0.0085">
  <pointmin x="-0.05" y="-0.05" z="-0.05" />
  <pointmax x="2.0" y="1.0" z="1.0" />
</definition>
```

Esto crea un dominio de 2.05m × 1.05m × 1.05m. El margen negativo (-0.05) es un colchon para que las particulas del borde no se escapen inmediatamente.

**¿Por que importa?** Porque el volumen del dominio × dp³ = numero total de particulas = tiempo de computo. Un dominio mas grande → mas particulas → mas lento.

### 3.3 Las Partes del Escenario

Todo dentro del dominio se clasifica en tipos de particulas usando **markers** (`mk`):

#### Boundary Particles (mk = mkbound)
Son las "paredes" solidas. No se mueven (a menos que les asignes movimiento). Forman:
- El **suelo** del canal
- Las **paredes** laterales
- El **boulder** (si, la roca tambien es boundary particles)

#### Fluid Particles (mk = mkfluid)
Son el **agua**. Se mueven libremente segun la fisica SPH. Al inicio estan quietas en la columna de dam break, y cuando empieza la simulacion, la gravedad las jala y generan el flujo.

#### Floating Particles
Son boundary particles especiales: las del boulder. La diferencia es que DualSPHysics calcula las fuerzas que el fluido ejerce sobre ellas y les permite moverse en respuesta (6 grados de libertad: 3 traslaciones + 3 rotaciones).

### 3.4 El Dam Break (Como se genera el "tsunami")

No se simula un tsunami real del oceano (eso requeriria un dominio de kilometros). Se usa un **dam break**: una columna de agua retenida por una pared invisible que desaparece en t=0.

```
ANTES (t=0):                    DESPUES (t>0):

████│                           ████
████│                           ████→→→→
████│                           ████→→→→→→→        ⬡
████│          ⬡                ████→→→→→→→→→→    ⬡⬡⬡
████│         ⬡⬡⬡               ████→→→→→→→→→→→  ⬡⬡⬡⬡
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓           ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
     ↑ pared invisible
       (desaparece)
```

La altura de la columna de agua controla la **energia** del flujo. Mas alta → mas velocidad → mas fuerza sobre el boulder. Esta es una de las variables parametricas de tu estudio.

---

## PARTE 4: EL BOULDER EN DUALSPHYSICS

### 4.1 De Roca Real a Particulas

El camino de una roca real a la simulacion:

```
Roca real → Escaneo 3D / Modelo CAD → Archivo .stl → DualSPHysics → Particulas
  (costa)     (fotogrametria)           (triangulos)    (GenCase)      (esferas)
```

Un archivo **STL** es una malla de triangulos que describe la **superficie** de un objeto 3D. No tiene informacion de volumen ni masa — solo caras.

### 4.2 El Problema del Hollow Shell (Bloque Hueco)

Cuando DualSPHysics importa un STL, solo crea particulas **en la superficie**:

```
Roca real:              STL importado sin llenar:      STL con fillbox:

  ██████                    ○○○○○○                      ○○○○○○
  ██████                    ○    ○                      ○●●●●○
  ██████                    ○    ○    ← HUECO           ○●●●●○  ← LLENO
  ██████                    ○    ○                      ○●●●●○
  ██████                    ○○○○○○                      ○○○○○○

  Masa correcta             Masa ~20% de la real        Masa correcta
  (solido, denso)           (cascara, como plumavit)    (despues de fillbox)
```

Sin el relleno:
- La roca pesa una fraccion de lo que deberia
- El agua la manda a volar instantaneamente
- Los resultados son fisicamente absurdos

**Solucion:** `fillbox` + `modefill void` (llena todo el vacio interior con particulas) + `massbody` (fuerza la masa real en kg).

### 4.3 Los 6 Grados de Libertad (DOF)

Un cuerpo rigido en 3D puede moverse de 6 formas:

```
TRASLACIONES (movimiento lineal):          ROTACIONES (giro):

  Surge (X) → adelante/atras               Roll  → giro sobre eje X (ladeo)
  Sway  (Y) → izquierda/derecha            Pitch → giro sobre eje Y (cabeceo)
  Heave (Z) → arriba/abajo                 Yaw   → giro sobre eje Z (guinada)

       z (Heave)                                  Yaw ↻
       │                                           │
       │    y (Sway)                               │
       │   ╱                                       │
       │  ╱                              Roll ↻────┼────→ Pitch ↻
       │ ╱                                         │
       └──────── x (Surge)
```

FloatingInfo te da la posicion del centro de masa + las 3 traslaciones (surge, sway, heave) + las 3 rotaciones (roll, pitch, yaw) en cada instante de tiempo. Con eso determinas si la roca se movio o no.

### 4.4 `mkbound` — El Identificador del Boulder

Cada grupo de particulas tiene un **marker** (`mk`) que lo identifica. Es como una etiqueta de color:

```xml
<setmkbound mk="0" />    <!-- Todo lo que dibujes ahora es mk=0 (suelo, paredes) -->
<drawbox> ... </drawbox>

<setmkbound mk="1" />    <!-- Todo lo que dibujes ahora es mk=1 (boulder) -->
<drawfilestl file="boulder.stl" />
```

Despues, cuando le dices a FloatingInfo "dame los datos del boulder", le pasas su mk:
```
FloatingInfo -onlymk:1 -savemotion -savecsv output
```

Y cuando defines el floating body:
```xml
<floating mkbound="1">          <!-- "las particulas con mk=1 son un cuerpo flotante" -->
  <massbody value="150.0" />    <!-- su masa real -->
</floating>
```

### 4.5 `massbody` vs `rhopbody`

Dos formas de decirle a DualSPHysics cuanto pesa el boulder:

- **`massbody`**: "Esta roca pesa **150 kg**." Directo, sin ambiguedad.
- **`rhopbody`**: "Esta roca tiene densidad **2600 kg/m³**." DualSPHysics calcula la masa multiplicando densidad × volumen_de_particulas. Pero el volumen de particulas NO es exactamente el volumen real del STL (la discretizacion introduce error). Resultado: masa incorrecta.

**Siempre usar `massbody`.** El volumen real lo calculas con trimesh en Python (`mesh.volume`), multiplicas por la densidad conocida de la roca, y metes el resultado como massbody.

### 4.6 Tensor de Inercia

Asi como la masa resiste el movimiento lineal, el **tensor de inercia** resiste la rotacion. Es una matriz 3×3:

```
         │ Ixx  Ixy  Ixz │
Inercia = │ Iyx  Iyy  Iyz │
         │ Izx  Izy  Izz │
```

- `Ixx` = resistencia a girar sobre el eje X (roll)
- `Iyy` = resistencia a girar sobre el eje Y (pitch)
- `Izz` = resistencia a girar sobre el eje Z (yaw)
- Los terminos cruzados (Ixy, etc.) son cero si los ejes pasan por el centro de masa y estan alineados con las direcciones principales del objeto. Para rocas irregulares, no seran cero.

`trimesh` calcula esto automaticamente: `mesh.moment_inertia` da la matriz 3×3. GenCase tambien la calcula desde las particulas, pero `trimesh` trabaja con la geometria continua y suele ser mas preciso.

---

## PARTE 5: LA CADENA DE EJECUCION (El "como" del software)

### 5.1 El Pipeline de DualSPHysics

```
   TU CODIGO PYTHON
         │
         ▼
  ┌──────────────┐     ┌─────────────────┐     ┌───────────────────┐
  │   GenCase     │────→│  DualSPHysics   │────→│  Post-Processing  │
  │              │     │                 │     │                   │
  │  XML + STL   │     │  .bi4 → .bi4   │     │  .bi4 → CSV       │
  │  → .bi4      │     │  (particulas    │     │  (FloatingInfo,   │
  │  (geometria) │     │   moviendose)   │     │   ComputeForces)  │
  └──────────────┘     └─────────────────┘     └───────────────────┘
     segundos            minutos-horas              segundos
```

### 5.2 GenCase — El Constructor

**Input:** `Case_Def.xml` (tu archivo de configuracion) + archivos STL
**Output:** `Case.bi4` (geometria inicial de particulas en binario)

GenCase lee el XML, interpreta todos los comandos de geometria (`drawbox`, `drawfilestl`, `fillbox`), y genera la disposicion inicial de particulas. Tambien calcula automaticamente `center` e `inertia` del floating body.

Es instantaneo (segundos). No usa GPU.

### 5.3 DualSPHysics — El Simulador

**Input:** `Case.bi4` (geometria inicial)
**Output:** `Part0000.bi4`, `Part0001.bi4`, ... (un snapshot por cada TimeOut)

Este es el nucleo. Toma las particulas iniciales y avanza la fisica paso a paso:

```
t=0.00s → Part0000.bi4  (agua quieta, boulder quieto)
t=0.01s → Part0001.bi4  (agua empieza a caer)
t=0.02s → Part0002.bi4  (agua avanza)
...
t=0.50s → Part0050.bi4  (agua impacta el boulder)
t=0.51s → Part0051.bi4  (¿se movio? ¿no se movio?)
...
t=2.00s → Part0200.bi4  (fin de la simulacion)
```

Cada Part.bi4 contiene la posicion y velocidad de **todas** las particulas en ese instante. Por eso los archivos son enormes.

**Flags criticos:**
- `-gpu` → usa la GPU (100x mas rapido que CPU)
- `-tmax:2.0` → simula 2 segundos de fisica
- `-tout:0.01` → guarda un snapshot cada 0.01 segundos (200 archivos para 2s)

### 5.4 FloatingInfo — El Espia del Boulder

**Input:** Los archivos Part.bi4
**Output:** Un CSV con la historia completa del boulder

No necesita releer todas las particulas del fluido. Solo le interesan las particulas del floating body (mk=1). Extrae:

| Columna | Que es | Unidades |
|---------|--------|----------|
| time | Instante de la simulacion | segundos |
| center X,Y,Z | Posicion del centro de masa | metros |
| fvel | Velocidad lineal del boulder | m/s |
| fomega | Velocidad angular del boulder | rad/s |
| surge | Desplazamiento en X acumulado | metros |
| sway | Desplazamiento en Y acumulado | metros |
| heave | Desplazamiento en Z acumulado | metros |
| roll | Rotacion sobre X acumulada | grados |
| pitch | Rotacion sobre Y acumulada | grados |
| yaw | Rotacion sobre Z acumulada | grados |

**Este CSV es el "oro" de cada simulacion.** Todo lo demas (.bi4, .vtk) se borra despues.

### 5.5 ComputeForces — El Medidor de Fuerza

Similar a FloatingInfo pero mide las **fuerzas** que el fluido ejerce sobre el boulder:

```
ComputeForces -dirin output -onlymk:1 -viscoart:0.1 -savecsv forces
```

Genera un CSV con fuerza (Fx, Fy, Fz) y momento (Mx, My, Mz) en cada instante. Util para la "Ley de Falla" que busca tu tesis: ¿a que fuerza critica se mueve la roca?

### 5.6 MeasureTool — El Sensor del Flujo

**NO mide el boulder.** Mide el **agua** en puntos fijos del espacio. Como poner sensores virtuales en el canal:

```
         sensor 1        sensor 2        sensor 3
            │                │                │
            ▼                ▼                ▼
  ████
  ████→→→→→→→→→→→→→→→→→→→→→→→→→→→→→        ⬡
  ████→→→→→→→→→→→→→→→→→→→→→→→→→→→→→      ⬡⬡⬡
  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
```

Cada sensor mide velocidad y presion del agua que pasa por ese punto. Util para caracterizar el flujo (¿a que velocidad llega el agua al boulder?), pero no para saber si el boulder se movio.

---

## PARTE 6: EL XML — El Plano de Construccion

### 6.1 Estructura General

El XML de DualSPHysics es como un plano de arquitectura dividido en secciones:

```xml
<case>
  <casedef>                    <!-- DEFINICION: que vas a simular -->
    <constantsdef> ... </constantsdef>    <!-- Fisica: gravedad, densidad, CFL -->
    <mkconfig> ... </mkconfig>             <!-- Cuantos tipos de particulas -->
    <geometry> ... </geometry>             <!-- Geometria: tanque, agua, boulder -->
    <floatings> ... </floatings>           <!-- Cuerpos flotantes: masa, inercia -->
  </casedef>

  <execution>                  <!-- EJECUCION: como simular -->
    <parameters> ... </parameters>         <!-- Algoritmo, viscosidad, tiempo -->
  </execution>
</case>
```

### 6.2 `constantsdef` — Las Leyes Fisicas

```xml
<constantsdef>
  <gravity x="0" y="0" z="-9.81" />    <!-- Gravedad (siempre -9.81 en Z) -->
  <rhop0 value="1000" />                <!-- Densidad del agua: 1000 kg/m³ -->
  <cflnumber value="0.2" />             <!-- Condicion CFL (estabilidad temporal) -->
  <coefh value="1.0" />                 <!-- Multiplicador del smoothing length -->
  <gamma value="7" />                   <!-- Exponente de la ecuacion de estado -->
  <coefsound value="20" />              <!-- Velocidad del sonido numerica -->
</constantsdef>
```

`gamma` y `coefsound` son parametros de la **ecuacion de estado** que relaciona presion con densidad. En SPH, el fluido es "debilmente compresible" (no 100% incompresible como el agua real). Estos valores estan estandarizados; no los tocas a menos que sepas exactamente por que.

### 6.3 `geometry` — El Escenario

Los comandos van dentro de `<commands><mainlist>`:

```xml
<geometry>
  <definition dp="0.0085">                 <!-- dp en metros -->
    <pointmin x="-0.05" y="-0.05" z="-0.05" />   <!-- esquina inferior del universo -->
    <pointmax x="2.0" y="1.0" z="1.0" />          <!-- esquina superior del universo -->
  </definition>
  <commands>
    <mainlist>
      <!-- 1. Dibujar paredes del tanque -->
      <setmkbound mk="0" />               <!-- etiqueta: mk=0 = paredes -->
      <drawbox>
        <boxfill>bottom|left|right|front|back</boxfill>  <!-- todas las caras menos el techo -->
        <point x="0" y="0" z="0" />
        <size x="1.6" y="0.67" z="0.4" />
      </drawbox>

      <!-- 2. Llenar la columna de agua -->
      <setmkfluid mk="0" />               <!-- etiqueta: mk=0 = agua -->
      <fillbox x="0.1" y="0.3" z="0.1">   <!-- seed point (dentro del agua) -->
        <modefill>void</modefill>
        <point x="0" y="0" z="0" />
        <size x="0.4" y="0.67" z="0.3" />
      </fillbox>

      <!-- 3. Importar el boulder -->
      <setmkbound mk="1" />               <!-- etiqueta: mk=1 = boulder -->
      <drawfilestl file="boulder.stl">
        <drawscale x="0.001" y="0.001" z="0.001" />  <!-- si el STL esta en mm -->
        <drawmove x="1.0" y="0.33" z="0.05" />       <!-- posicionar en el canal -->
      </drawfilestl>

      <!-- 4. Rellenar el boulder (fix hollow shell) -->
      <fillbox x="1.0" y="0.33" z="0.1">   <!-- seed = dentro del boulder -->
        <modefill>void</modefill>
        <point x="0.8" y="0.1" z="0.0" />
        <size x="0.4" y="0.5" z="0.3" />
      </fillbox>
    </mainlist>
  </commands>
</geometry>
```

### 6.4 `floatings` — Decir que el Boulder se Puede Mover

Sin esta seccion, el boulder seria una pared fija inamovible:

```xml
<floatings>
  <floating mkbound="1">                        <!-- las particulas mk=1 son un flotante -->
    <massbody value="15.0" />                    <!-- masa real: 15 kg -->
    <center x="1.0" y="0.33" z="0.1" />         <!-- centro de gravedad -->
    <inertia>                                    <!-- tensor de inercia 3x3 -->
      <values v11="0.05" v12="0" v13="0" />
      <values v21="0" v22="0.08" v23="0" />
      <values v31="0" v32="0" v33="0.06" />
    </inertia>
  </floating>
</floatings>
```

### 6.5 `execution > parameters` — Como Simular

```xml
<execution>
  <parameters>
    <parameter key="StepAlgorithm" value="2" />     <!-- 2=Symplectic (mejor) -->
    <parameter key="Kernel" value="2" />             <!-- 2=Wendland (mejor para flotantes) -->
    <parameter key="ViscoTreatment" value="1" />     <!-- 1=Artificial -->
    <parameter key="Visco" value="0.1" />            <!-- Valor de viscosidad -->
    <parameter key="DensityDT" value="2" />          <!-- 2=Fourtakas (recomendado) -->
    <parameter key="RigidAlgorithm" value="1" />     <!-- Como mover el boulder -->
    <parameter key="FtPause" value="0.5" />          <!-- Congelar boulder 0.5s al inicio -->
    <parameter key="TimeMax" value="2.0" />          <!-- Simular 2 segundos -->
    <parameter key="TimeOut" value="0.01" />         <!-- Guardar cada 0.01s -->
  </parameters>
</execution>
```

**`FtPause`** merece atencion: si pones 0.5, el boulder esta "congelado" los primeros 0.5 segundos. El agua cae y se asienta, y recien despues el boulder es libre de moverse. Esto evita artefactos numericos al inicio.

**`RigidAlgorithm`** controla como se calcula la interaccion del boulder:
- **0** = sin colisiones (boulder flota en el agua, no choca con el suelo)
- **1** = SPH nativo (colisiones basicas con el suelo via particulas boundary)
- **2** = DEM (Discrete Element Method — colisiones mas realistas)
- **3** = Chrono (motor de fisica multi-cuerpo — el mas sofisticado, permite friccion real)

Si Diego usa Chrono (valor 3), el XML tiene secciones adicionales. Si usa 1, es mas simple.

---

## PARTE 7: EL CRITERIO DE FALLA — ¿Se movio o no?

### 7.1 El Problema

Despues de cada simulacion, FloatingInfo te da un CSV con 200 filas (una por cada 0.01s). Necesitas una respuesta binaria: **¿se movio?** Si/No.

Pero "moverse" no esta definido con precision. ¿Si se desplaza 0.001mm es movimiento? ¿Si vibra pero vuelve a su lugar? ¿Si solo rota 0.1 grados?

### 7.2 La Propuesta (pendiente de validacion con Dr. Moris)

```
MOVIMIENTO = SI  si se cumple CUALQUIERA de:
  1. Desplazamiento del centro de masa > X% del diametro equivalente
  2. Rotacion neta > Y grados

MOVIMIENTO = NO  si ninguna se cumple al final de la simulacion
```

Donde:
- **Diametro equivalente** = el diametro de una esfera con el mismo volumen que el boulder: `d_eq = (6V/π)^(1/3)`
- **X%** = ¿5%? ¿10%? → lo decide el Dr. Moris
- **Y grados** = ¿5°? ¿10°? → lo decide el Dr. Moris

Tambien se registra **t_crit**: el instante exacto en que se cruza el umbral.

### 7.3 Por que esto es importante academicamente

Las formulas empiricas (Nandasena, etc.) usan criterios diferentes y a menudo subjetivos. Definir un criterio **numerico, reproducible y automatizado** es parte de la contribucion de tu tesis. El Dr. Moris tiene que validarlo antes de que lo implementes.

---

## PARTE 8: EL ESTUDIO PARAMETRICO — La Vision Completa

### 8.1 ¿Que Estamos Buscando?

La **frontera de estabilidad**: la linea que separa "la roca se mueve" de "la roca no se mueve" en el espacio de variables.

```
  Altura de ola (h)
       │
   2.0 │  ×  ×  ×  ×  ×       × = se movio
       │  ×  ×  ×  ×  ○       ○ = no se movio
   1.5 │  ×  ×  ×  ○  ○
       │  ×  ×  ○  ○  ○       La linea entre × y ○
   1.0 │  ×  ○  ○  ○  ○       es la FRONTERA DE ESTABILIDAD
       │  ○  ○  ○  ○  ○
   0.5 │  ○  ○  ○  ○  ○
       └────────────────── Masa del boulder (M)
         10   50  100  500  1000 kg
```

Pero no son solo 2 variables. Son multiples dimensiones: masa, altura de ola, angulo, forma, densidad, friccion. Por eso se usa Latin Hypercube Sampling para explorar eficientemente.

### 8.2 Latin Hypercube Sampling (LHS)

Imagina que tienes 2 variables y quieres probar 5 combinaciones. Grid search probaria una cuadricula:

```
Grid (25 sims para cubrir bien):    LHS (5 sims con buena cobertura):

  h │ × × × × ×                     h │ ×
    │ × × × × ×                       │       ×
    │ × × × × ×                       │   ×
    │ × × × × ×                       │         ×
    │ × × × × ×                       │     ×
    └──────────── M                    └──────────── M
```

LHS distribuye los puntos de forma que cada "fila" y "columna" del espacio tiene exactamente un punto. Cobertura maxima con minimas simulaciones. Con 4-7 variables, LHS es dramaticamente mas eficiente que grid search.

### 8.3 El Surrogate Model (ML)

Con 100-500 simulaciones NO puedes cubrir todo el espacio de variables. Pero SI puedes entrenar un modelo de Machine Learning que **interpole** entre los puntos simulados:

```
Datos de simulacion (puntos)     →     Gaussian Process (superficie continua)

  h │  ×          ○                   h │ ████████░░░░
    │     ×    ○                       │ █████░░░░░░░
    │  ×    ○  ○                       │ ███░░░░░░░░░   █ = se mueve
    │    ○   ○                         │ █░░░░░░░░░░░   ░ = no se mueve
    │ ○    ○                           │ ░░░░░░░░░░░░
    └──────────── M                    └──────────── M
```

El GP no solo predice si/no, sino que da **intervalos de confianza**: "estoy 95% seguro de que aqui se mueve" vs "estoy 60% seguro — necesito mas datos aqui". Eso permite **active learning**: las proximas simulaciones se hacen donde la incertidumbre es mayor.

### 8.4 Monte Carlo sobre el Surrogate (Cuantificacion de Incertidumbre)

Hasta aqui, cada simulacion SPH produce un numero: "con esta masa y esta altura de ola, el boulder se desplazo 1.6m". Pero en la realidad, esos parametros tienen **incertidumbre**: la masa de una roca no se conoce con precision exacta, la altura del tsunami varia, la friccion depende de la rugosidad local.

**El problema:** Propagar esa incertidumbre a traves de la simulacion SPH directamente es imposible. Si cada run toma 4 horas, correr 10,000 variaciones tomaria 4.5 anos.

**La solucion:** Usar el GP surrogate como reemplazo instantaneo de la simulacion. El GP predice en milisegundos lo que DualSPHysics toma horas en calcular. Entonces:

```
1. Entrenar GP con ~50 simulaciones reales (semanas de computo)
2. Definir incertidumbre de cada parametro:
     masa = 150 kg ± 10%  →  distribucion Normal(150, 15)
     altura = 0.3m ± 20%  →  distribucion Normal(0.3, 0.06)
     friccion = 0.5 ± 30% →  distribucion Normal(0.5, 0.15)
3. Generar 10,000 combinaciones aleatorias de esos parametros (Monte Carlo)
4. Evaluar CADA combinacion en el GP (10 segundos total)
5. Resultado: distribucion de desplazamiento, no un solo numero
```

```
Sin UQ:                          Con UQ (Monte Carlo + GP):

"El boulder se desplaza 1.6m"   "El boulder se desplaza 1.6m ± 0.3m (95% CI)"
                                 "Probabilidad de movimiento incipiente: 87%"
                                 "El parametro que mas afecta: altura de ola (62%)"
```

Esto es **Uncertainty Quantification (UQ)**: convertir un resultado puntual en una distribucion con confianza estadistica. La tecnica se llama **Monte Carlo sobre surrogate** y es estandar en ingenieria computacional (Oakley & O'Hagan, 2004; Saltelli et al., 2008).

### 8.5 Analisis de Sensibilidad (Indices de Sobol)

Una vez que tienes el GP entrenado, puedes responder la pregunta mas importante para el ingeniero: **¿que parametro importa mas?**

Los **indices de Sobol** descomponen la varianza total del resultado en contribuciones de cada parametro de entrada:

```
Varianza total del desplazamiento = 100%

  Altura de ola (h):     62%  ████████████████████████████████
  Masa del boulder (M):  23%  ████████████
  Angulo de playa:        8%  ████
  Friccion:               5%  ███
  Interacciones:          2%  █
```

Esto te dice:
- **Primer orden (S_i):** Efecto directo de cada parametro aislado
- **Orden total (ST_i):** Efecto del parametro incluyendo interacciones con otros

Si el indice de Sobol de la altura es 0.62, significa que el 62% de la incertidumbre en el resultado viene de no conocer la altura exacta. Implicacion practica: si quieres reducir la incertidumbre de tu prediccion, mide mejor la altura de ola — mejorar la precision de la masa o la friccion aporta mucho menos.

El calculo de indices de Sobol sobre un GP surrogate es computacionalmente trivial (Sudret, 2008; Sobol, 2001). Se hace con muestreo Monte Carlo de Saltelli (Saltelli, 2002): generas matrices de muestras, evaluas el GP miles de veces, y calculas las varianzas condicionales.

### 8.6 El Pipeline Completo: Simulacion → Surrogate → UQ

El flujo completo conecta todo lo anterior:

```
FASE 1: DATOS (WS, semanas)
  LHS genera 50 combinaciones de parametros
  → DualSPHysics corre 50 simulaciones GPU
  → ETL extrae metricas de cada run
  → Tabla: [masa, altura, angulo, ...] → [desplaz, rotacion, fuerza]

FASE 2: SURROGATE (laptop, minutos)
  Entrenar GP sobre los 50 puntos
  → Validar con leave-one-out cross-validation
  → Verificar: error < 5% en prediccion

FASE 3: UQ (laptop, segundos)
  Monte Carlo: 10,000 muestras aleatorias → evaluar GP
  → Distribucion de desplazamiento con intervalos de confianza
  → Indices de Sobol: que parametro domina
  → Frontera de estabilidad probabilistica (no determinista)

FASE 4: DECISION (tesis)
  "La probabilidad de movimiento incipiente es >95% cuando h > 0.35m
   para un boulder de 150 kg con forma BLIR3. El parametro dominante
   es la altura de ola (S_T = 0.62). La masa es secundaria (S_T = 0.23)."
```

Este pipeline tiene precedente directo en la literatura: Salmanidou et al. (2017, 2020) usaron exactamente este enfoque (simulacion SPH de tsunami + emulador GP + indices de Sobol) para cuantificar incertidumbre en oleaje por deslizamiento de tierra. La diferencia es que tu tesis aplica el metodo al **transporte de bloques costeros con geometria irregular**, lo cual no se ha hecho.

**Referencia clave:** Salmanidou, D.M., Heidarzadeh, M., & Guillas, S. (2020). "Uncertainty Quantification of Landslide Generated Waves Using Gaussian Process Emulation and Variance-Based Sensitivity Analysis." *Water*, 12(2), 416. DOI: 10.3390/w12020416 — paper open-access que describe casi exactamente tu pipeline.

### 8.7 ¿Cuantas Simulaciones se Necesitan?

Regla practica de Loeppky et al. (2009): para entrenar un GP, necesitas **~10 × d** puntos, donde d es el numero de variables de entrada. Con 5 variables (masa, altura, angulo, friccion, forma) → minimo 50 simulaciones.

Si cada simulacion toma ~4 horas en la WS (dp=0.004, RTX 5090):
- 50 simulaciones = ~200 horas GPU = ~8 dias continuos
- O ~12 dias con pausas y monitoreo

Esto es factible. Y una vez entrenado el GP, el Monte Carlo/UQ es instantaneo.

---

## PARTE 9: REFERENCIAS CLAVE

### SPH y DualSPHysics
- Crespo, A.J.C. et al. (2015). "DualSPHysics: Open-source parallel CFD solver based on SPH." *Computer Physics Communications*, 187, 204-216. DOI: 10.1016/j.cpc.2014.10.004
- Dominguez, J.M. et al. (2022). "DualSPHysics: from fluid dynamics to multiphysics problems." *Computational Particle Mechanics*, 9(5), 867-895. DOI: 10.1007/s40571-021-00404-2
- Lind, S.J. et al. (2020). "Review of smoothed particle hydrodynamics: towards converged Lagrangian flow modelling." *Proceedings of the Royal Society A*, 476(2241). DOI: 10.1098/rspa.2019.0801
- Canelas, R.B. et al. (2016). "SPH-DCDEM model for arbitrary geometries in free surface solid-fluid flows." *Computer Physics Communications*, 202, 131-140. DOI: 10.1016/j.cpc.2016.01.006

### Transporte de Bloques por Tsunami
- Nandasena, N.A.K. et al. (2011). "Reassessment of hydrodynamic equations: Minimum flow velocity to initiate boulder transport." *Marine Geology*, 281(1-4), 70-84. DOI: 10.1016/j.margeo.2011.02.005
- Imamura, F. et al. (2008). "A numerical model for the transport of a boulder by tsunami." *JGR: Oceans*, 113, C01008. DOI: 10.1029/2007JC004170
- Goto, K. et al. (2014). "Boulder transport by the 2011 Great East Japan tsunami." *Marine Geology*, 346, 292-309. DOI: 10.1016/j.margeo.2013.09.015
- Oetjen, J. et al. (2021). "Experiments on tsunami induced boulder transport: A review." *Earth-Science Reviews*, 220, 103714. DOI: 10.1016/j.earscirev.2021.103714

### Gaussian Process / Surrogate Modeling
- Rasmussen, C.E. & Williams, C.K.I. (2006). *Gaussian Processes for Machine Learning.* MIT Press. Disponible: gaussianprocess.org/gpml
- Sacks, J. et al. (1989). "Design and Analysis of Computer Experiments." *Statistical Science*, 4(4), 409-423. DOI: 10.1214/ss/1177012413
- Forrester, A.I.J. et al. (2008). *Engineering Design via Surrogate Modelling: A Practical Guide.* Wiley. DOI: 10.1002/9780470770801
- Loeppky, J.L. et al. (2009). "Choosing the Sample Size of a Computer Experiment." *Technometrics*, 51(4), 366-376. DOI: 10.1198/TECH.2009.08040
- Bastos, L.S. & O'Hagan, A. (2009). "Diagnostics for Gaussian Process Emulators." *Technometrics*, 51(4), 425-438. DOI: 10.1198/TECH.2009.08019

### Latin Hypercube Sampling
- McKay, M.D. et al. (1979). "A Comparison of Three Methods for Selecting Values of Input Variables." *Technometrics*, 21(2), 239-245. DOI: 10.1080/00401706.1979.10489755
- Helton, J.C. & Davis, F.J. (2003). "Latin hypercube sampling and the propagation of uncertainty." *Reliability Engineering & System Safety*, 81(1), 23-69. DOI: 10.1016/S0951-8320(03)00058-9

### Cuantificacion de Incertidumbre (UQ)
- Oakley, J.E. & O'Hagan, A. (2004). "Probabilistic sensitivity analysis of complex models: a Bayesian approach." *JRSS-B*, 66(3), 751-769. DOI: 10.1111/j.1467-9868.2004.05304.x
- Salmanidou, D.M. et al. (2017). "Statistical emulation of landslide-induced tsunamis." *Proceedings of the Royal Society A*, 473(2200). DOI: 10.1098/rspa.2017.0026
- **Salmanidou, D.M. et al. (2020). "Uncertainty Quantification of Landslide Generated Waves Using GP Emulation and Variance-Based Sensitivity Analysis." *Water*, 12(2), 416. DOI: 10.3390/w12020416** ← paper mas cercano a tu metodologia
- Guillas, S. et al. (2018). "Functional emulation of high resolution tsunami modelling over Cascadia." *Annals of Applied Statistics*, 12(4), 2023-2053. DOI: 10.1214/18-AOAS1142

### Analisis de Sensibilidad (Sobol)
- Sobol, I.M. (2001). "Global sensitivity indices for nonlinear mathematical models." *Mathematics and Computers in Simulation*, 55(1-3), 271-280. DOI: 10.1016/S0378-4754(00)00270-6
- Saltelli, A. (2002). "Making best use of model evaluations to compute sensitivity indices." *Computer Physics Communications*, 145(2), 280-297. DOI: 10.1016/S0010-4655(02)00280-1
- Saltelli, A. et al. (2008). *Global Sensitivity Analysis: The Primer.* Wiley. DOI: 10.1002/9780470725184
- Sudret, B. (2008). "Global sensitivity analysis using polynomial chaos expansions." *RESS*, 93(7), 964-979. DOI: 10.1016/j.ress.2007.04.002

---

## PARTE 10: GLOSARIO RAPIDO

| Termino | Que es | Analogia |
|---------|--------|----------|
| `dp` | Distancia entre particulas | Pixeles de la simulacion |
| `h` | Radio de influencia de cada particula | Alcance de la "vista" de cada particula |
| CFL | Condicion de estabilidad temporal | Limite de velocidad para que nada se rompa |
| `pointmin/pointmax` | Esquinas del dominio | Las paredes del universo simulado |
| `mk` / marker | Etiqueta de tipo de particula | Color que identifica cada grupo |
| `mkbound` | Marker de particulas solidas | "Soy pared" o "soy boulder" |
| `mkfluid` | Marker de particulas fluidas | "Soy agua" |
| `fillbox` | Comando para llenar una region | "Rellena este espacio con particulas" |
| `modefill void` | Llenar solo los vacios encerrados | "Rellena el interior del STL" |
| `massbody` | Masa real del floating body | "Esta roca pesa X kg" |
| `floatings` | Seccion que declara cuerpos moviles | "Estas particulas pueden moverse" |
| `FtPause` | Congelar flotantes N segundos | "Espera que el agua se asiente" |
| Dam Break | Columna de agua que colapsa | "Tsunami en miniatura" |
| Surge/Sway/Heave | Traslacion en X/Y/Z | Movimiento lineal de la roca |
| Roll/Pitch/Yaw | Rotacion sobre X/Y/Z | Giro de la roca |
| FloatingInfo | Herramienta que extrae cinematica | "El espia que vigila la roca" |
| ComputeForces | Herramienta que extrae fuerzas | "El medidor de fuerza sobre la roca" |
| GenCase | Pre-procesador (XML → particulas) | "El constructor del escenario" |
| Part.bi4 | Snapshot de todas las particulas en un instante | "Foto de todas las particulas en t=X" |
| Convergencia de malla | Verificar que dp es suficientemente fino | "Verificar que los pixeles son suficientes" |
| LHS | Latin Hypercube Sampling | "Elegir combinaciones inteligentemente" |
| GP | Gaussian Process (surrogate model) | "IA que interpola entre simulaciones" |
| $t_{crit}$ | Instante critico de movimiento | "El milisegundo en que la roca se rinde" |
| Monte Carlo | Repetir con variaciones aleatorias para medir estabilidad | "Lanzar la pelota 10,000 veces y ver donde cae" |
| UQ | Uncertainty Quantification — ponerle barras de error a los resultados | "No es 1.6m, es 1.6m ± 0.3m" |
| Surrogate | Modelo ML que reemplaza la simulacion costosa | "Atajo instantaneo para predecir sin simular" |
| Sobol (indices) | Medida de cuanto contribuye cada parametro a la incertidumbre total | "La altura de ola explica el 62% de la variacion" |
| Sensibilidad (S_i) | Efecto directo de un parametro sobre el resultado | "Si cambio solo la masa, cuanto cambia el desplazamiento" |
| Orden total (ST_i) | Efecto de un parametro incluyendo interacciones con otros | "Masa sola + masa×altura + masa×friccion + ..." |
| Cross-validation (LOO) | Validar el GP dejando 1 punto fuera y viendo si lo predice bien | "Tapar un dato, predecirlo, y ver cuanto te equivocaste" |
| Active learning | Usar la incertidumbre del GP para decidir donde simular siguiente | "Simular donde el GP tiene mas duda" |
| CI (Confidence Interval) | Rango donde cae el resultado con X% de probabilidad | "Entre 1.3m y 1.9m con 95% de confianza" |
