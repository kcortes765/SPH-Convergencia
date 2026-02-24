# PLAN.md — SPH-IncipientMotion: Plan Maestro de Implementación

> **Estado:** Pre-implementación (19 febrero 2026)
> **Autor:** Kevin Cortés Hernández
> **Asesor:** Dr. Joaquin Moris Barra (UCN)
>
> Este documento distingue explícitamente entre lo que **SABEMOS** (verificado), lo que **ASUMIMOS** (probable pero no confirmado), y lo que **DESCONOCEMOS** (bloqueante hasta obtener información).

---

## 1. OBJETIVO DEL PROYECTO

**SABEMOS:**
- El objetivo científico es determinar umbrales de movimiento incipiente de bloques costeros (boulders) ante flujos tipo tsunami, usando simulación SPH.
- El capstone se enmarca en un Fondecyt de Iniciación del Dr. Moris (ANID).
- Se busca una relación matemática entre velocidad del flujo (u), altura de ola (h), masa del bloque (M) y forma geométrica.
- Las fórmulas empíricas existentes (Nandasena, Engel & May) simplifican excesivamente al tratar los bloques como prismas rectangulares homogéneos.
- La innovación es incorporar la **forma irregular real** del bloque en las ecuaciones predictivas.

**ASUMIMOS:**
- Que el enfoque Dam Break (rotura de presa) es el método principal para generar el flujo tipo tsunami en la simulación. Esto es estándar en la literatura de DualSPHysics, pero el Dr. Moris podría preferir otro generador de ola (pistón, flap, solitary wave).

**DESCONOCEMOS:**
- Las condiciones de contorno exactas que el Dr. Moris quiere replicar (dimensiones del canal virtual, condiciones iniciales del flujo).
- Si hay datos experimentales de laboratorio (del Fondecyt general del Dr. Moris) contra los cuales validar. **El capstone de Kevin es 100% numérico — no incluye trabajo de laboratorio físico.** El Dr. Moris tiene experimentos de lab en su proyecto mayor, pero son independientes.
- El rango de variables a explorar (masas mínima/máxima, alturas de ola, ángulos de inclinación, cuántas formas de boulder).

---

## 2. MOTOR DE SIMULACIÓN: DualSPHysics

### 2.1 Versión

**SABEMOS:**
- La documentación del proyecto menciona DualSPHysics v5.2+.
- La última versión **estable** es **v5.4.3** (lanzada marzo 2025).
- Existe una **v6.0 beta** (principios de 2026) — **NO USAR** en la tesis para evitar bugs no documentados.
- Los ejecutables cambian de nombre según la versión (`DualSPHysics5.2_win64.exe` vs `DualSPHysics5.4_win64.exe`).

**DECISIÓN TOMADA:** Usar la versión que tenga Diego (5.2 o 5.4.x). No migrar a v6.0 beta bajo ninguna circunstancia.

### 2.2 Cadena de Ejecución (Verificado vía documentación oficial)

```
GenCase → DualSPHysics (GPU) → PostProcessing (FloatingInfo / MeasureTool / ComputeForces / PartVTK)
```

**Comandos CLI reales (verificados):**

```bash
# 1. Pre-procesamiento: XML → geometría de partículas (.bi4)
GenCase Case_Def Case [options]
  -dp:[float]           # distancia entre partículas (CLAVE)
  -save:+/-all,+/-bi,+/-vtkall,+/-vtkbound,+/-vtkfluid

# 2. Simulación SPH
DualSPHysics Case [options]
  -gpu[:id]             # modo GPU (obligatorio para producción)
  -tmax:[float]         # tiempo máximo de simulación (s)
  -tout:[float]         # intervalo de escritura de output (s)
  -sv:binx              # formato de salida (binx por defecto)
  -symplectic           # algoritmo de integración temporal
  -viscoart:[float]     # viscosidad artificial (rango 0-1, típico 0.1)
  -deltasph:[float]     # constante Delta-SPH (típico 0.1)
  -shifting:[mode]      # corrección de shifting (none/nobound/nofixed/full)
  -posdouble:[0/1/2]    # precisión de posición

# 3. Post-procesamiento: extraer datos del floating body
FloatingInfo -onlymk:[mk] -savemotion -savedata OutputName
  → Genera: posición del centro de masa, velocidad lineal,
    velocidad angular, surge/sway/heave, pitch/roll/yaw

# 4. Post-procesamiento alternativo: fuerzas sobre el bloque
ComputeForces -onlymk:[mk] -savecsv OutputName
  → Genera: fuerzas hidrodinámicas sobre el cuerpo

# 5. Post-procesamiento: mediciones en puntos fijos
MeasureTool -dirin CaseOut -points points.txt -vars:-all,+vel,+press -savecsv OutputName
  → Genera: velocidad y presión interpoladas en puntos definidos

# 6. Visualización (opcional, solo para validación visual)
PartVTK -onlymk:[mk] -vars:+vel,+press -savevtk OutputVTK
```

**IMPORTANTE — Descubrimiento:** Existe la herramienta `FloatingInfo` que es **más directa** que MeasureTool para obtener el movimiento del bloque. FloatingInfo da directamente posición del centro de masa, velocidades y rotaciones del cuerpo flotante. MeasureTool interpola variables en puntos fijos del espacio (útil para medir el flujo, no el bloque).

### 2.3 Estructura del XML de Caso (Verificado)

```xml
<case>
  <casedef>
    <constantsdef>
      <gravity x="0" y="0" z="-9.81" />
      <rhop0 value="1000" />           <!-- densidad de referencia del fluido -->
      <cflnumber value="0.2" />         <!-- condición CFL -->
      <coefh value="1.0" />             <!-- coeficiente del smoothing length -->
    </constantsdef>

    <mkconfig boundcount="240" fluidcount="10" />

    <geometry>
      <definition dp="0.0085">          <!-- distancia entre partículas (m) -->
        <pointmin x="..." y="..." z="..." />
        <pointmax x="..." y="..." z="..." />
      </definition>
      <commands>
        <!-- Límites del tanque -->
        <setmkbound mk="0" />
        <drawbox> ... </drawbox>

        <!-- Fluido (Dam Break) -->
        <setmkfluid mk="0" />
        <fillbox> ... </fillbox>

        <!-- Boulder importado desde STL -->
        <setmkbound mk="1" />
        <drawfilestl file="boulder.stl">
          <drawscale x="1" y="1" z="1" />
        </drawfilestl>

        <!-- RELLENO del boulder (solución hollow shell) -->
        <fillbox x="[cx]" y="[cy]" z="[cz]">    <!-- seed = centroide -->
          <modefill>void</modefill>
          <point x="[xmin]" y="[ymin]" z="[zmin]" />
          <size x="[dx]" y="[dy]" z="[dz]" />
        </fillbox>
      </commands>
    </geometry>

    <!-- Definición del cuerpo flotante -->
    <floatings>
      <floating mkbound="1">
        <massbody value="[masa_real_kg]" />     <!-- OBLIGATORIO: masa real, no calculada -->
        <center x="[cx]" y="[cy]" z="[cz]" />  <!-- centro de gravedad -->
        <inertia>                                <!-- tensor de inercia -->
          <values v11="..." v12="0" v13="0" />
          <values v21="0" v22="..." v23="0" />
          <values v31="0" v32="0" v33="..." />
        </inertia>
      </floating>
    </floatings>
  </casedef>

  <execution>
    <parameters>
      <parameter key="StepAlgorithm" value="2" />     <!-- 1=Verlet, 2=Symplectic -->
      <parameter key="Kernel" value="2" />             <!-- 1=Cubic, 2=Wendland -->
      <parameter key="ViscoTreatment" value="1" />     <!-- 1=Artificial -->
      <parameter key="Visco" value="0.1" />
      <parameter key="DensityDT" value="2" />          <!-- 0=None, 1=Molteni, 2=Fourtakas -->
      <parameter key="RigidAlgorithm" value="1" />    <!-- 0=collision-free, 1=SPH, 2=DEM, 3=Chrono -->
      <parameter key="FtPause" value="0.0" />          <!-- Congelar flotantes N seg (útil para asentar fluido) -->
      <parameter key="TimeMax" value="[tiempo_max]" />
      <parameter key="TimeOut" value="[intervalo]" />
    </parameters>
  </execution>
</case>
```

**ASUMIMOS:**
- Que el caso de Diego sigue una estructura similar. Las etiquetas son estándar de DualSPHysics, pero la versión específica podría tener diferencias menores.
- Que el bloque se define como `floating` (libre de moverse). Si fuera `moving` o `fixed`, la lógica cambia.

**DESCONOCEMOS:**
- El `dp` (resolución) que usa Diego. Esto define la calidad y el costo computacional.
- Si Diego usa Chrono (motor de rigid body más avanzado) o el solver nativo de floating bodies.
- Los parámetros exactos de fricción suelo-bloque.
- Si hay condiciones de contorno periódicas o paredes especiales.

### 2.4 El Problema del Hollow Shell (Solución Verificada)

**SABEMOS con certeza:**
- Los archivos STL solo contienen información de caras (superficie), no volumen.
- `drawfilestl` solo crea una cáscara de partículas boundary.
- La solución oficial es agregar un `<fillbox>` con `<modefill>void</modefill>` dentro del bloque `drawfilestl`.
- El `fillbox` necesita: un **seed point** (punto interior al STL), y un **tamaño** que cubra todo el STL.
- Se DEBE usar `<massbody>` para fijar la masa real del objeto, porque la discretización por partículas introduce errores en masa/inercia.

**RIESGOS TÉCNICOS IDENTIFICADOS:**
- En los foros de DualSPHysics, usuarios reportan que `fillbox` a veces no llena completamente STLs complejos si la malla tiene defectos (holes, non-manifold edges). Se necesita validar visualmente en ParaView después de cada GenCase que el bloque esté correctamente lleno.
- Si el llenado falla, posible causa: normales del STL invertidas. Fix en Blender: Edit Mode → F3 → "Flip Normals" → re-exportar.
- `autofill="true"` en `<drawfilestl>` es poco confiable para geometría compleja — intenta llenar desde el centro de volumen que podría estar vacío. Mejor usar `fillbox` explícito.
- El seed point del `fillbox` debe estar a **mínimo 2h de distancia** de las partículas boundary (h = smoothing length = `coefh * sqrt(3 * dp²)` en 3D).

---

## 3. ARQUITECTURA DEL PIPELINE PYTHON

### 3.1 Visión General

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  MÓDULO 1       │     │  MÓDULO 2    │     │  MÓDULO 3   │     │  MÓDULO 4    │
│  Geometry       │────→│  Batch       │────→│  ETL /      │────→│  ML          │
│  Builder        │     │  Runner      │     │  Data Clean │     │  Surrogate   │
│                 │     │              │     │             │     │              │
│  STL + Template │     │  GenCase →   │     │  FloatingInfo│    │  Gaussian    │
│  → N XMLs       │     │  DSPH → Post │     │  CSV → SQL  │     │  Process     │
└─────────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
        │                                            │                    │
        ▼                                            ▼                    ▼
  config/cases/*.xml                         data/results.sqlite    Frontera de
  (generados)                                (tabla maestra)        Estabilidad
```

### 3.2 Módulo 1 — Geometry Builder

**Responsabilidad:** Generar N archivos XML de caso a partir de un template + STL + matriz de parámetros.

**Lo que SABEMOS implementar:**
- Leer STL con `trimesh`: calcular centro de masa (`mesh.center_mass`), bounding box (`mesh.bounds`), volumen (`mesh.volume`), momentos de inercia (`mesh.moment_inertia`).
- Parsear XML con `lxml.etree`: localizar las secciones de geometría y floatings.
- Inyectar: coordenadas del `fillbox` (seed = centro de masa, size = bounding box), `<massbody>`, `<center>`, `<inertia>`.
- Para rotaciones: `trimesh` soporta `mesh.apply_transform(rotation_matrix)` para rotar el STL antes de calcular propiedades. `scipy.spatial.transform.Rotation` para generar matrices de rotación a partir de ángulos de Euler.

**Lo que ASUMIMOS:**
- Que `trimesh` calcula correctamente el centro de masa para mallas cerradas (manifold). Si el STL de Diego tiene defectos, `trimesh` podría dar valores erróneos.
- Que el centro de masa de `trimesh` (geométrico, asumiendo densidad uniforme) es suficiente para el boulder. En la realidad, la densidad del boulder podría no ser uniforme, pero para la simulación SPH asumimos densidad homogénea.
- Que las rotaciones se aplican sobre el centroide del STL. Hay que verificar si DualSPHysics rota el STL al importarlo o si necesitamos rotar el archivo STL previamente.

**Lo que DESCONOCEMOS:**
- La estructura exacta del XML de Diego (qué etiquetas usa, dónde está el bloque definido, si usa `drawfilestl` o alguna otra forma).
- Si el STL del boulder es una malla cerrada y limpia (requisito para que `trimesh` calcule volumen e inercia).
- El sistema de coordenadas del STL (puede estar centrado en origen o en una esquina arbitraria).
- Si DualSPHysics acepta STLs rotados directamente o si hay que rotar las coordenadas del `fillbox` y `center` por separado. **NOTA:** La investigación confirma que `<drawfilestl>` acepta hijos `<drawscale>`, `<drawmove>` y `<drawrotate>` para transformar la geometría in situ, sin modificar el archivo STL original. Esto simplifica el pipeline (no hay que generar N STLs rotados, solo variar las transformaciones en el XML).
- Las unidades del STL (metros, milímetros, etc.). Si están en mm, se puede usar `<drawscale x="0.001" y="0.001" z="0.001" />` para convertir a metros.

**Variables paramétricas a variar (confirmadas conceptualmente):**

| Variable | Tipo | Rango | Fuente |
|----------|------|-------|--------|
| Masa del bloque (M) | Continua | **POR DEFINIR** | Dr. Moris |
| Altura de ola / nivel Dam Break (h) | Continua | **POR DEFINIR** | Dr. Moris |
| Ángulo de inclinación del bloque | Continua (0°-360°?) | **POR DEFINIR** | Diseño experimental |
| Forma geométrica | Categórica | **POR DEFINIR** (¿cuántos STLs?) | Dr. Moris |
| Densidad del bloque (ρ) | Continua | **POR DEFINIR** | Según tipo de roca |
| Coeficiente de fricción | Continua | **POR DEFINIR** | Calibración |
| Dp (resolución) | Fija por estudio | **POR DEFINIR** | Convergence study |

### 3.3 Módulo 2 — Batch Runner

**Responsabilidad:** Ejecutar la cadena GenCase → DualSPHysics → FloatingInfo/ComputeForces de forma desatendida para N casos.

**Lo que SABEMOS implementar:**
```python
import subprocess, pathlib, logging

def run_case(case_xml: pathlib.Path, dsph_bin: pathlib.Path, timeout_s: int):
    case_name = case_xml.stem.replace("_Def", "")
    out_dir = case_xml.parent / f"{case_name}_out"

    steps = [
        # GenCase
        [str(dsph_bin / "GenCase_win64.exe"), str(case_xml), str(out_dir / case_name)],
        # DualSPHysics GPU
        [str(dsph_bin / "DualSPHysics5.X_win64.exe"),  # ← versión exacta POR CONFIRMAR
         str(out_dir / case_name), str(out_dir),
         "-gpu", "-sv:binx", f"-tmax:{tmax}", f"-tout:{tout}"],
        # FloatingInfo
        [str(dsph_bin / "FloatingInfo_win64.exe"),
         "-dirin", str(out_dir), "-onlymk:1",   # ← mk del boulder POR CONFIRMAR
         "-savemotion", "-savecsv", str(out_dir / "floating_data")],
    ]

    for cmd in steps:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout_s)
        if result.returncode != 0:
            logging.error(f"FALLO: {cmd[0]} returncode={result.returncode}")
            logging.error(result.stderr.decode())
            return False
    return True
```

**Lo que ASUMIMOS:**
- Que los ejecutables de DualSPHysics se llaman `*_win64.exe` y están en un directorio conocido.
- Que la ejecución GPU es estable (no hay memory leaks que causen crash después de muchas simulaciones consecutivas).
- Que `FloatingInfo` es la herramienta correcta para extraer el movimiento del boulder (no MeasureTool).
- Que cada simulación termina en un tiempo razonable (minutos a horas, no días) para la resolución de desarrollo.

**Lo que DESCONOCEMOS:**
- El nombre exacto de los ejecutables de la versión que usa Diego.
- Si hay que configurar variables de entorno (PATH, CUDA_VISIBLE_DEVICES).
- El timeout apropiado por simulación (depende de dp y tmax).
- Si DualSPHysics escribe output a stdout/stderr de forma útil para logging.
- El consumo de VRAM por simulación y si se puede estimar a priori.
- Si la RTX 4060 (8GB) puede correr las simulaciones de desarrollo sin quedarse sin memoria.

**DECISIÓN PENDIENTE:** ¿Usar `FloatingInfo` solamente, o también `ComputeForces` para obtener las fuerzas hidrodinámicas sobre el bloque? Para la "Ley de Falla" podrían necesitarse ambos: FloatingInfo para cinemática (se movió o no) y ComputeForces para dinámica (qué fuerza causó el movimiento).

### 3.4 Módulo 3 — ETL / Data Cleaner

**Responsabilidad:** Leer las salidas CSV de FloatingInfo/ComputeForces, aplicar el criterio de falla, almacenar en SQLite.

**Lo que SABEMOS:**
- `FloatingInfo` con `-savemotion` genera CSV con columnas de: time, posición del centro (x,y,z), velocidad lineal (vx,vy,vz), velocidad angular (wx,wy,wz), surge/sway/heave, pitch/roll/yaw.
- DualSPHysics usa **punto y coma (`;`)** como separador CSV **por defecto** — esto está hardcodeado en el código fuente (clase `jcsv::JSaveCsv2` usa format strings como `"%g;%g;%g"`). El formato decimal es siempre **punto** (inglés), independiente del locale de Windows. El problema de Diego con comas decimales venía de abrir el CSV en Excel español, no del software.
- **CONFIRMADO:** Se puede forzar coma como separador con el flag `-csvsep:1` en la línea de comandos de las herramientas de post-proceso. Decisión: usar `-csvsep:1` en nuestros comandos para simplificar el parsing con `pd.read_csv(path, sep=',')`.
- Precisión por defecto: 6 dígitos significativos con formato `%g`.

**Criterio de Falla (definido conceptualmente, NO calibrado):**

```python
# Concepto del criterio — los umbrales exactos son POR DEFINIR
def evaluar_falla(df_floating, d_eq, umbral_desplazamiento=0.05, umbral_rotacion_deg=5.0):
    """
    df_floating: DataFrame con columnas de FloatingInfo (time, cx, cy, cz, ...)
    d_eq: diámetro equivalente del boulder (m)
    umbral_desplazamiento: fracción de d_eq (5% = 0.05)
    umbral_rotacion_deg: grados de rotación neta

    Retorna: (hubo_falla: bool, t_crit: float o None)
    """
    pos_inicial = df_floating.iloc[0][['cx', 'cy', 'cz']].values
    desplazamiento = np.linalg.norm(df_floating[['cx','cy','cz']].values - pos_inicial, axis=1)

    # Criterio cinemático: desplazamiento > 5% del diámetro equivalente
    falla_desp = desplazamiento > umbral_desplazamiento * d_eq

    # Criterio rotacional: rotación acumulada > umbral
    rotacion = np.sqrt(df_floating['pitch']**2 + df_floating['roll']**2 + df_floating['yaw']**2)
    falla_rot = rotacion > umbral_rotacion_deg

    falla = falla_desp | falla_rot
    if falla.any():
        t_crit = df_floating.loc[falla.idxmax(), 'time']
        return True, t_crit
    return False, None
```

**CONFIRMADO (verificado contra source code v5.4.x):** Las columnas de FloatingInfo son:
`time`, `center (X, Y, Z)`, `fvel`, `fomega`, `surge`, `sway`, `heave`, `roll`, `pitch`, `yaw`

**Lo que DESCONOCEMOS:**
- Si el CSV de la versión que usa Diego tiene exactamente estos headers o variaciones menores.
- Si el 5% del diámetro equivalente es un criterio válido para la defensa académica, o si el Dr. Moris tiene otro criterio en mente.
- Si necesitamos también un criterio de velocidad (v > v_threshold en un instante dado).
- El diámetro equivalente del boulder (se calcula desde el volumen del STL: `d_eq = (6V/π)^(1/3)`).

**Esquema de Base de Datos (SQLite):**

```sql
CREATE TABLE resultados (
    case_id         TEXT PRIMARY KEY,       -- nombre único del caso
    -- Parámetros de entrada
    masa_kg         REAL NOT NULL,
    densidad_kg_m3  REAL NOT NULL,
    altura_ola_m    REAL NOT NULL,
    angulo_x_deg    REAL NOT NULL,
    angulo_y_deg    REAL NOT NULL,
    angulo_z_deg    REAL NOT NULL,
    forma_id        TEXT NOT NULL,           -- identificador del STL usado
    dp_m            REAL NOT NULL,
    coef_friccion   REAL,
    -- Resultados
    hubo_movimiento INTEGER NOT NULL,        -- 0/1 booleano
    t_crit_s        REAL,                    -- NULL si no hubo movimiento
    desplaz_max_m   REAL,
    rotacion_max_deg REAL,
    fuerza_max_N    REAL,                    -- de ComputeForces (si se usa)
    -- Metadatos
    duracion_sim_s  REAL,                    -- tiempo de cómputo
    fecha_ejecucion TEXT,
    notas           TEXT
);
```

**ASUMIMOS:**
- Que SQLite es suficiente para el volumen esperado (500-1000 filas, no millones).
- Que no hay necesidad de acceso concurrente a la BD (las simulaciones corren en serie).

### 3.5 Módulo 4 — ML Surrogate Model

**Responsabilidad:** Entrenar un modelo predictivo sobre la tabla de resultados.

**Lo que SABEMOS:**
- `GaussianProcessRegressor` de scikit-learn es apropiado porque cuantifica incertidumbre (confidence intervals), que es lo que necesita un investigador para saber "dónde simular más".
- Los inputs son las variables paramétricas, el output es la probabilidad/magnitud de movimiento.
- Con 100-500 datos, un GP es razonable. Con más datos, habría que considerar sparse GPs o Random Forests.

**Lo que ASUMIMOS:**
- Que la "forma" del boulder se puede codificar como features numéricas (ratios de ejes del bounding box: a/b, b/c, esfericidad de Wadell, etc.) en lugar de usar la geometría completa.
- Que el kernel RBF o Matérn es apropiado para la física del problema.
- Que 100-500 simulaciones son suficientes para un surrogate model razonable.

**Lo que DESCONOCEMOS:**
- Si el Dr. Moris prefiere un modelo de clasificación (movimiento sí/no) o regresión (magnitud del desplazamiento / fuerza crítica).
- Si hay que comparar contra las fórmulas empíricas existentes (Nandasena, Engel & May) como baseline.
- El número mínimo de simulaciones necesarias para que el GP sea confiable (depende de la dimensionalidad del espacio de parámetros).
- Si la variable "forma" se puede representar de forma suficientemente rica con descriptores geométricos simples.

**ESTE MÓDULO ES EL ÚLTIMO EN IMPLEMENTARSE.** No tiene sentido trabajar en ML hasta que el pipeline de datos funcione.

---

## 4. DISEÑO DE EXPERIMENTOS

### 4.1 Latin Hypercube Sampling (LHS)

**SABEMOS:**
- LHS es superior a grid search para explorar espacios de alta dimensión con budget limitado.
- `scipy.stats.qmc.LatinHypercube` o `pyDOE2.lhs()` generan la matriz.
- Cada fila de la matriz es una combinación de parámetros que se traduce en un XML.

**ASUMIMOS:**
- Que las variables son independientes (no hay correlaciones impuestas entre masa y forma, por ejemplo).
- Que 100 simulaciones iniciales son un buen punto de partida para un primer surrogate.

**DESCONOCEMOS:**
- El número total de variables (dimensiones del espacio). Si son 4 (masa, altura, ángulo, forma) con ~100 puntos está bien. Si son 7+ (agregar densidad, fricción, dp), necesitamos más puntos.
- Los rangos de cada variable (mínimo, máximo). Sin esto, no se puede generar la matriz.
- Si el Dr. Moris quiere un diseño factorial completo para algún subconjunto de variables.

### 4.2 Estudio de Convergencia de Malla (OBLIGATORIO antes del barrido paramétrico)

**SABEMOS que hay que hacer esto pero NO está en el plan original de Gemini:**
- Antes de correr 500 simulaciones, hay que verificar que los resultados son independientes de `dp`.
- Se toma UN caso representativo y se corre con dp = [grueso, medio, fino, muy fino].
- Se compara la fuerza sobre el bloque / desplazamiento entre resoluciones.
- Cuando la diferencia entre dos resoluciones consecutivas es < 5%, se elige la más gruesa como "dp de producción".
- Esto es **crítico para la defensa académica** — sin convergence study, los evaluadores pueden rechazar todos los resultados.

**SOBRE LA RESOLUCIÓN DEL BOULDER:** La documentación de DualSPHysics NO establece un mínimo de partículas por dimensión del floating body. Lo que sí está verificado es la regla **H/dp ≈ 10** (10 partículas por altura de ola) como referencia para la resolución del flujo ([fuente](https://dual.sphysics.org/faq/)). Para el boulder, la resolución adecuada se determina empíricamente con el **estudio de convergencia** — no con una regla a priori. A menor dp, mejor se captura la forma irregular, pero el costo crece cúbicamente.

**DESCONOCEMOS:**
- El tamaño real del boulder de Diego (afecta cuántas partículas lo representan para un dp dado).
- Las dimensiones del dominio virtual (canal/tanque — determina el conteo total de partículas).
- El tiempo de cómputo por resolución.
- Si la RTX 4060 puede correr la resolución de producción o solo las de desarrollo.

---

## 5. GESTIÓN DE ARCHIVOS Y ALMACENAMIENTO

**SABEMOS:**
- Cada simulación de DualSPHysics genera archivos `.bi4` (binarios de partículas) por cada timestep. Tamaño por snapshot ≈ `N_partículas × 32 bytes` (posición xyz + velocidad xyz + densidad + ID). Ejemplos concretos:
  - 1M partículas, tout=0.01s, 2s simulación → ~200 snapshots × 32 MB = **~6.4 GB**
  - 10M partículas, misma config → **~64 GB**
  - 20M partículas → **~128 GB** por simulación
- Los archivos VTK convertidos son significativamente más grandes que los `.bi4`.
- El pipeline DEBE borrar los `.bi4` y `.vtk` después de extraer las métricas con FloatingInfo/ComputeForces.
- Solo se conservan: el XML de caso, los CSV de FloatingInfo/ComputeForces, y la BD SQLite.

**ASUMIMOS:**
- Que el disco de la laptop tiene al menos 500 GB libres para desarrollo (correr pocas simulaciones a la vez).
- Que la workstation con RTX 5090 tiene almacenamiento suficiente para el batch de producción.
- Que borrar los `.bi4` es seguro una vez extraídas las métricas (no hay necesidad de re-analizar los datos crudos).

**RIESGO CRÍTICO (confirmado por revisión externa):** Si el Batch Runner falla en borrar los `.bi4` después de procesarlos, el disco de la RTX 5090 (probablemente 1-2 TB) se llena en un solo fin de semana de batch processing.

**Mandato arquitectónico:** La limpieza de `.bi4` DEBE estar en un bloque `try/finally` a prueba de balas. Incluso si la simulación falla, el post-proceso se interrumpe, o hay un error inesperado, el script debe garantizar la limpieza del disco. Borrar solo DESPUÉS de verificar que los CSV de FloatingInfo/ComputeForces existen y tienen datos válidos.

```python
# Patrón obligatorio para el batch runner
def run_and_clean(case_dir):
    try:
        run_simulation(case_dir)
        extract_metrics(case_dir)  # FloatingInfo → CSV
        validate_outputs(case_dir) # verificar que CSV existe y tiene filas
    finally:
        cleanup_binaries(case_dir) # borrar .bi4, .vtk SIEMPRE
```

**Estimaciones de tiempo de cómputo (verificadas con benchmarks de la comunidad):**

| Partículas | GPU | Tiempo físico | Wall-clock aprox |
|-----------|-----|--------------|-----------------|
| ~100K | RTX 4060 | 2s | minutos |
| 1M | RTX 4060 | 2s | ~30 min |
| 5M | RTX 4060 (8GB) | 2s | 3-6 horas |
| 10M | RTX 5090 (32GB) | 2s | 2-4 horas |
| 20M+ | RTX 5090 (32GB) | 2s | 6-12+ horas |

Estos tiempos son **estimados** y dependen de dp, kernel, viscosidad, y parámetros específicos.

---

## 6. CRONOGRAMA REVISADO

El cronograma original (4 sprints en febrero) tiene el **1 de marzo** como hito. Estamos a **19 de febrero**. Análisis realista:

### Sprint 0 — Prerequisitos (BLOQUEANTE)
**Estado: EN ESPERA de archivos de Diego**

- [ ] Obtener de Diego: `template_base.xml`, `boulder.stl`, `run_command.bat`, `sample_output.csv`
- [ ] Instalar DualSPHysics en la laptop (confirmar versión)
- [ ] Correr manualmente el caso de Diego para verificar que funciona
- [ ] Examinar el XML, el STL, el BAT y el CSV reales
- [ ] Confirmar con Dr. Moris: variables de estudio, rangos, criterio de falla

**Sin estos archivos, NO se puede avanzar con código que interactúe con DualSPHysics.** Se puede avanzar con la estructura del proyecto, el esqueleto de los módulos (con stubs), y el diseño de la BD.

### Sprint 1 — Fundación (Cuando se tengan los archivos)

- [ ] Crear estructura de directorios del proyecto
- [ ] Leer el XML real y mapear la estructura (¿dónde están las etiquetas a modificar?)
- [ ] Leer el STL real con `trimesh` y validar: ¿es manifold? ¿volumen correcto? ¿unidades?
- [ ] Script básico: `geometry_builder.py` que lea el STL, calcule centro de masa e inercia, e inyecte en el XML template
- [ ] Validación: correr GenCase con el XML generado, visualizar en ParaView, confirmar relleno correcto

### Sprint 2 — Ejecución Automatizada

- [ ] `batch_runner.py`: ejecutar GenCase → DualSPHysics → FloatingInfo para UN caso
- [ ] Logging y manejo de errores (timeout, crash)
- [ ] Parsear el CSV de FloatingInfo, confirmar nombres de columnas
- [ ] `data_cleaner.py`: leer CSV, aplicar criterio de falla, escribir en SQLite
- [ ] Test end-to-end: generar 1 XML → simular → extraer → almacenar resultado

### Sprint 3 — Escala

- [ ] Generador de matriz experimental (LHS) → N XMLs
- [ ] Loop batch: correr N simulaciones en serie con limpieza automática de `.bi4`
- [ ] Estudio de convergencia de malla (dp)
- [ ] `main_orchestrator.py`: unifica todo
- [ ] Smoke test: 5-10 casos completos end-to-end

### Sprint 4 — ML y Presentación (post-hito de marzo, probablemente)

- [ ] `ml_surrogate.py`: entrenar GP sobre datos reales
- [ ] Validación GP: Leave-One-Out cross-validation, error < 5%
- [ ] Monte Carlo sobre GP: 10,000 muestras → distribuciones + CI 95%
- [ ] Índices de Sobol: identificar parámetros dominantes
- [ ] Frontera de estabilidad probabilística (no determinista)
- [ ] Visualización de la frontera de estabilidad
- [ ] Comparación con fórmulas empíricas (Nandasena, etc.)
- [ ] Documentación para el Dr. Moris

---

## 7. DECISIONES TÉCNICAS POR TOMAR

| # | Decisión | Opciones | Impacto | Quién decide |
|---|----------|----------|---------|-------------|
| 1 | Versión DualSPHysics | 5.2 vs 5.4 | Compatibilidad con archivos de Diego | Diego / Dr. Moris |
| 2 | Herramienta de post-proceso principal | FloatingInfo vs MeasureTool vs ambos | Define qué datos extraemos | Técnico (verificar con caso real) |
| 3 | ¿Usar ComputeForces además de FloatingInfo? | Sí (fuerzas + cinemática) vs No (solo cinemática) | Define si podemos construir la "Ley de Falla" en fuerzas | Dr. Moris |
| 4 | Criterio de falla: umbrales | 5% desplazamiento / 5° rotación / velocidad / otro | Define el output binario del ETL | Dr. Moris (académico) |
| 5 | Tipo de generación de ola | Dam Break vs Pistón vs Solitary wave | Cambia la geometría del XML | Dr. Moris |
| 6 | ¿Usar Chrono para rigid body dynamics? | Chrono (más preciso, colisiones) vs nativo (más simple) | Complejidad del XML y del setup | Técnico + Dr. Moris |
| 7 | Variables del estudio paramétrico | {M, h, θ, forma} vs más variables | Dimensionalidad del LHS | Dr. Moris |
| 8 | Modelo ML | GP (con incertidumbre) vs Random Forest vs SVM | Tipo de predicción y visualización | Técnico + Dr. Moris |
| 9 | Codificación de la "forma" para ML | Ratios de ejes / esfericidad / Fourier descriptors | Riqueza de la representación | Técnico |
| 10 | Dp de producción | Valor exacto (ej: 0.005, 0.002, ...) | Costo computacional vs precisión | Convergence study |

---

## 8. RIESGOS Y MITIGACIONES

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|-------------|---------|-----------|
| Diego no entrega archivos a tiempo | Media | **CRÍTICO** — bloquea todo | Contactar directamente al Dr. Moris. En paralelo, usar los ejemplos de DualSPHysics (CaseDambreak, CaseSolidsCHRONO) como casos de desarrollo |
| STL del boulder tiene defectos de malla | Media | Alto — `fillbox` no llena correctamente | Reparar con `trimesh.repair` o MeshLab antes de usar. Validar siempre visualmente |
| RTX 4060 sin VRAM suficiente para simulaciones de desarrollo | Baja | Medio — hay que reducir dp o tamaño del dominio | Usar dp grueso para desarrollo, reservar resolución fina para la RTX 5090 |
| El criterio de falla del 5% no es aceptado académicamente | Media | Alto — invalida el análisis | Definir el criterio CON el Dr. Moris antes de implementar. Revisar literatura (Nandasena usa otros criterios) |
| DualSPHysics crashea en simulaciones largas (memory leak GPU) | Baja | Medio — pierde tiempo de cómputo | Watchdog con restart desde checkpoint: `DualSPHysics -partbegin:69 dirbegin case out_case` (retoma desde Part0069.bi4) |
| El GP no converge con 100-500 datos | Media | Medio — necesita más simulaciones | Usar active learning (GP selecciona los próximos puntos a simular donde la incertidumbre es mayor) |
| El formato CSV de FloatingInfo cambia entre versiones | Baja | Bajo — hay que ajustar el parser | Hacer el parser flexible (detección automática de columnas) |

---

## 9. STACK TECNOLÓGICO CONFIRMADO

| Componente | Tecnología | Certeza |
|------------|-----------|---------|
| Simulación CFD | DualSPHysics (v5.2 o v5.4) | CONFIRMADO |
| Lenguaje de automatización | Python 3.10+ | CONFIRMADO |
| Lectura STL | `trimesh` | CONFIRMADO (mejor alternativa: `numpy-stl` si trimesh falla) |
| Manipulación XML | `lxml.etree` | CONFIRMADO |
| Procesamiento de datos | `pandas` + `numpy` | CONFIRMADO |
| Rotaciones 3D | `scipy.spatial.transform.Rotation` | CONFIRMADO |
| Diseño experimental | `scipy.stats.qmc.LatinHypercube` | CONFIRMADO |
| Base de datos | SQLite via `sqlalchemy` o `sqlite3` | CONFIRMADO |
| Machine Learning | `scikit-learn` (GaussianProcessRegressor) | PROBABLE (podría cambiar) |
| Visualización de resultados | `matplotlib` / `plotly` | PROBABLE |
| Dashboard interactivo | Streamlit o Power BI | POR DECIDIR |
| Gestión de entorno | `venv` o `uv` | POR DECIDIR |
| Visualización CFD | ParaView (manual, para validación) | CONFIRMADO |

---

## 10. ARCHIVOS QUE SE PUEDEN CREAR HOY (sin archivos de Diego)

Mientras esperamos los archivos de Diego, podemos avanzar en:

1. **Estructura de directorios** del proyecto
2. **`requirements.txt`** o `pyproject.toml` con las dependencias
3. **Esqueleto de cada módulo** (funciones con docstrings y `raise NotImplementedError`)
4. **Esquema SQLite** (crear la tabla `resultados`)
5. **Script LHS** (generador de matriz experimental — solo necesita los rangos, que se configuran después)
6. **Script de utilidades** (`utils.py`: logging, paths, configuración)
7. **Este mismo PLAN.md** como documentación viva

**NO se puede crear aún:**
- El parser XML real (no conocemos la estructura del template de Diego)
- El builder de geometría real (no conocemos el STL ni cómo interactúa con el XML)
- El batch runner con comandos reales (no conocemos los nombres exactos de los .exe)
- El parser de CSV de FloatingInfo (no conocemos los headers)

---

## 11. PREGUNTAS PARA EL DR. MORIS (siguiente reunión)

1. ¿Qué versión de DualSPHysics usa Diego? ¿Hay posibilidad de migrar a v5.4?
2. ¿Cuáles son las dimensiones del canal numérico (largo × ancho × alto)?
3. ¿El flujo se genera con Dam Break, pistón, o solitary wave?
4. ¿Cuántos STLs de boulder diferentes hay? ¿Son reales (escaneados) o generados?
5. ¿Cuál es el rango de masas a estudiar? ¿Y de alturas de ola?
6. ¿Cómo define usted "movimiento incipiente"? ¿Hay un criterio en la literatura que prefiera?
7. ¿Necesitamos extraer fuerzas (ComputeForces) además de cinemática (FloatingInfo)?
8. ¿Hay datos experimentales de laboratorio contra los cuales validar las simulaciones?
9. ¿Usa Diego el módulo Chrono para la interacción del cuerpo rígido?
10. ¿El estudio de convergencia de malla (dp) ya fue realizado por Diego, o hay que hacerlo desde cero?

---

## 12. RESUMEN EJECUTIVO

**Documentación de referencia:** `EDUCACION_TECNICA.md` contiene explicaciones detalladas de toda la física, el método SPH, los parámetros, la estructura XML y el flujo completo del proyecto.

**Lo que tenemos:** Un plan de arquitectura sólido con 4 módulos bien definidos, entendimiento profundo de DualSPHysics (XML, CLI, herramientas de post-proceso), y un stack tecnológico verificado.

**Lo que nos falta para empezar a codificar en serio:** Los 4 archivos base de Diego (XML, STL, BAT, CSV) y las respuestas del Dr. Moris sobre las variables del estudio.

**Lo que podemos hacer mientras tanto:** Crear la estructura del proyecto, los esqueletos de los módulos, el esquema de BD, el generador LHS, y practicar con los casos de ejemplo de DualSPHysics (01_DamBreak, 07_DambreakCubes).

**El riesgo principal:** Que los archivos de Diego se retrasen y el hito del 1 de marzo quede comprometido. Mitigación: usar los ejemplos oficiales de DualSPHysics como stand-in temporal para desarrollar el pipeline.

---

## 13. HERRAMIENTA ALTERNATIVA: DesignSPHysics

**Descubrimiento de la investigación:** Existe [DesignSPHysics](https://github.com/DualSPHysics/DesignSPHysics), un plugin oficial de FreeCAD escrito en Python que genera XMLs de DualSPHysics desde una GUI. Soporta: geometría, floating bodies, fill boxes, wave generators, ejecución y post-procesamiento.

**Relevancia para el proyecto:**
- **NO lo usaremos como herramienta principal** (necesitamos automatización headless, no GUI).
- **SÍ es útil como referencia** — su código Python muestra exactamente cómo generar XML de DualSPHysics programáticamente. Si hay dudas sobre cómo estructurar una etiqueta XML, podemos inspeccionar el código de DesignSPHysics como documentación viva.

---

## 14. FUENTES VERIFICADAS

- [DualSPHysics Wiki — Running](https://github.com/DualSPHysics/DualSPHysics/wiki/5.-Running-DualSPHysics)
- [DualSPHysics Wiki — Testcases](https://github.com/DualSPHysics/DualSPHysics/wiki/7.-Testcases)
- [DualSPHysics FAQ](https://dual.sphysics.org/faq/)
- [Foro: Filling STL with boundary particles](https://forums.dual.sphysics.org/discussion/812/filling-an-imported-stl-with-boundary-particles)
- [Foro: Filling STL using fill box](https://forums.dual.sphysics.org/discussion/2652/filling-stl-file-using-fill-box)
- [Foro: XML guide for rigid objects](https://forums.dual.sphysics.org/discussion/1628/xml-guide-definition-of-attributes-of-rigid-objects)
- [Foro: Floating objects in DualSPHysics](https://forums.dual.sphysics.org/discussion/25/floating-objects-in-dualsphysics)
- [Foro: Floating object mass body](https://forums.dual.sphysics.org/discussion/2186/floating-object-mass-body-gives-a-wrong-value)
- [Foro: CSV gauge format/precision](https://forums.dual.sphysics.org/discussion/1928/gauges-change-the-number-format-and-precision-in-the-output-file-code-directions)
- [GitHub: CaseDambreak_Def.xml](https://github.com/DualSPHysics/DualSPHysics/blob/master/examples/main/01_DamBreak/CaseDambreak_Def.xml)
- [GitHub: DesignSPHysics](https://github.com/DualSPHysics/DesignSPHysics)
- [MDPI: Flow-Debris Interaction DualSPHysics-CHRONO](https://www.mdpi.com/2076-3417/11/8/3618)
- [MDPI: Dam-Break Wave with Sharp Obstacle](https://www.mdpi.com/2073-4441/13/15/2133)
