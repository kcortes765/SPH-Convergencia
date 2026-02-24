# Capítulo 3: Metodología Numérica

## 3.1 Introducción

El presente capítulo describe la metodología numérica empleada para simular la interacción entre un flujo tipo tsunami y un bloque costero irregular mediante Hidrodinámica de Partículas Suavizadas (SPH). Se detalla la configuración del dominio computacional, las propiedades del bloque, los parámetros del solver, la integración con el motor de cuerpos rígidos ProjectChrono, y el estudio de convergencia de malla que valida la resolución espacial seleccionada para la campaña paramétrica.

Todas las simulaciones fueron realizadas con DualSPHysics v5.4.355 (GenCase v5.4.354.01), utilizando aceleración GPU y acoplamiento Chrono para la dinámica del cuerpo rígido.

---

## 3.2 Dominio Computacional

### 3.2.1 Geometría del Canal

El dominio computacional reproduce un canal con playa inclinada, diseñado para generar un flujo tipo dam-break que impacta un bloque posicionado en la zona de rompiente.

**Dimensiones del dominio:**
- Longitud: 15.0 m
- Ancho: 1.0 m
- Altura: 1.55 m
- Límites: x ∈ [-0.05, 15.05], y ∈ [-0.05, 1.05], z ∈ [-0.15, 1.55]

**Geometría del fondo:**
La playa se modela mediante un archivo STL (`Canal_Playa_1esa20_750cm.stl`) con pendiente 1:20, representativo de un perfil costero suave. Las paredes laterales, frontal y trasera se generan como partículas de contorno (boundary) tipo `drawbox` con relleno `bottom|left|right|front|back`.

### 3.2.2 Columna de Agua (Dam-Break)

La condición inicial del flujo se genera mediante una columna de agua estática que colapsa por gravedad en t = 0, produciendo una onda tipo bore que se propaga hacia la playa.

**Parámetros de la columna de agua:**
- Altura: 0.3 m (configurable en rango [0.2, 0.5] m)
- Longitud: 3.0 m
- Ancho: completo del canal (1.0 m)
- Generación: `fillbox` con semilla interior y modo `void`

Esta configuración tipo dam-break es ampliamente utilizada en la literatura para simular flujos tipo tsunami a escala de laboratorio (Noji et al., 1993; Imamura et al., 2008).

### 3.2.3 Instrumentación Virtual

Se distribuyeron 20 sensores virtuales (gauges) a lo largo del canal:

| Tipo | Cantidad | Variable | Intervalo de muestreo |
|------|----------|----------|----------------------|
| Velocidad | 12 (V01-V12) | Velocidad del flujo (m/s) | 0.001 s |
| Altura máxima | 8 (hmax01-hmax08) | Elevación máxima del agua (m) | 0.001 s |

Los gauges se posicionan automáticamente a distancias incrementales desde la posición del bloque, permitiendo caracterizar el campo de velocidades y alturas del flujo incidente.

---

## 3.3 Bloque Costero

### 3.3.1 Geometría

Se emplea un bloque irregular digitalizado mediante escaneo 3D (modelo BLIR3), representativo de un canto rodado costero.

**Propiedades geométricas (escaladas ×0.04):**

| Propiedad | Valor | Unidad |
|-----------|-------|--------|
| Dimensiones (bbox) | 17.1 × 21.0 × 4.0 | cm |
| Volumen | 0.530 | L (5.30 × 10⁻⁴ m³) |
| Diámetro equivalente (d_eq) | 10.04 | cm |
| Vértices / Caras | 1,198 / 2,392 | — |
| Estanqueidad (watertight) | Sí | — |

El diámetro equivalente se calcula como:

$$d_{eq} = \left(\frac{6V}{\pi}\right)^{1/3}$$

### 3.3.2 Propiedades Físicas

Las propiedades de masa e inercia se calculan a partir de la geometría STL usando la librería `trimesh`, que integra sobre la malla triangular cerrada para obtener valores exactos del sólido continuo.

| Propiedad | Valor | Método |
|-----------|-------|--------|
| Masa | 1.061 kg | Explícita (`massbody`) |
| Densidad implícita | 2000 kg/m³ | M/V desde trimesh |
| Centro de masa (local) | (-0.020, 0.012, 0.018) m | trimesh |
| Inercia Ixx | 0.00219 kg·m² | trimesh |
| Inercia Iyy | 0.00158 kg·m² | trimesh |
| Inercia Izz | 0.00361 kg·m² | trimesh |

**Corrección crítica:** Se identificó que GenCase sobreestima la inercia en un factor de 1.85× a 3.01× cuando se usa dp grueso (dp = 0.05 m), debido a que la discretización SPH agranda el volumen efectivo del cuerpo. Por esta razón, se inyectan los valores de inercia calculados por trimesh directamente en el XML mediante la etiqueta `<inertia>`, en lugar de usar los valores auto-calculados por GenCase.

**Posición en el dominio:**
- Posición inicial: (8.5, 0.5, 0.1) m
- Rotación inicial: configurable (0°, 0°, 0°) por defecto

### 3.3.3 Problema del Cuerpo Hueco

La importación de geometría STL mediante `drawfilestl` en DualSPHysics genera únicamente partículas en la superficie del cuerpo, dejando el interior vacío. Un bloque hueco tiene un comportamiento dinámico radicalmente diferente a un sólido (menor masa efectiva, inercia incorrecta).

**Solución implementada:** Después de importar la superficie STL, se aplica un `fillbox` con modo `void` cuya semilla se ubica en el centroide del mesh (calculado por trimesh). Esto rellena el interior con partículas de contorno, produciendo un sólido completo. Adicionalmente, se especifica la masa real del cuerpo mediante `massbody` para evitar errores de discretización volumétrica.

---

## 3.4 Parámetros del Solver SPH

### 3.4.1 Configuración General

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| Kernel | Wendland (C2) | Estándar para cuerpos flotantes en SPH |
| Integración temporal | Symplectic (2° orden) | Mayor precisión que Verlet |
| Viscosidad | Artificial, α = 0.05 | Estabiliza sin excesiva disipación |
| Difusión de densidad | Fourtakas (DDT tipo 2), δ = 0.1 | Reduce ruido en campo de presión |
| CFL | 0.2 | Conservador, garantiza estabilidad |
| Coeficiente h | 0.75 | h = 0.75 × √(3 × dp²) |
| Precisión posicional | Doble (`posdouble:1`) | Necesario para dominios > 10 m |

### 3.4.2 Dinámica del Cuerpo Rígido (ProjectChrono)

La interacción entre el fluido SPH y el bloque rígido se resuelve mediante el acoplamiento con ProjectChrono (`RigidAlgorithm = 3`), que maneja:

- Dinámica de cuerpo rígido (traslación y rotación)
- Detección y resolución de colisiones
- Fuerzas de fricción en el contacto bloque-fondo

**Parámetros de Chrono:**

| Parámetro | Valor |
|-----------|-------|
| Método de contacto | NSC (Non-Smooth Contacts) |
| Margen de colisión | 0.5 × dp |
| Intervalo de guardado CSV | 0.001 s |
| Cuerpos | BLIR (flotante), beach (fijo) |

**Propiedades de material (PVC/Acero — configuración de laboratorio):**

| Material | Módulo Young (Pa) | Poisson | Restitución | Fricción cinética |
|----------|-------------------|---------|-------------|-------------------|
| PVC (bloque) | 3.0 × 10⁹ | 0.30 | 0.60 | 0.15 |
| Acero (canal) | 2.1 × 10¹¹ | 0.35 | 0.80 | 0.35 |

**Nota:** Los valores de fricción corresponden a una configuración de laboratorio (PVC sobre acero). Para escenarios de campo, se deberán ajustar a materiales tipo caliza (μ ≈ 0.35-0.50).

### 3.4.3 Tiempo de Asentamiento (FtPause)

Se implementa un período de congelamiento de 0.5 s (`FtPause = 0.5`) durante el cual el bloque permanece inmóvil. Esto permite que:

1. El fluido se asiente bajo gravedad
2. Las presiones hidrostáticas se estabilicen
3. Se eliminen perturbaciones numéricas iniciales

Sin este período, el bloque experimenta aceleraciones espurias al inicio de la simulación que no tienen origen físico.

---

## 3.5 Datos de Salida

### 3.5.1 Cinemática del Bloque (ChronoExchange)

El archivo `ChronoExchange_mkbound_51.csv` se genera automáticamente durante la simulación y contiene, para cada paso de tiempo (Δt = 0.001 s):

| Variable | Columnas | Unidades |
|----------|----------|----------|
| Posición centro de masa | fcenter.x, .y, .z | m |
| Velocidad lineal | fvel.x, .y, .z | m/s |
| Velocidad angular | fomega.x, .y, .z | rad/s |
| Aceleración lineal | face.x, .y, .z | m/s² |
| Aceleración angular | fomegaace.x, .y, .z | rad/s² |

### 3.5.2 Fuerzas sobre el Bloque (ChronoBody_forces)

El archivo `ChronoBody_forces.csv` registra las fuerzas hidrodinámicas SPH y las fuerzas de contacto Chrono por separado:

| Componente | Variables | Unidades |
|------------|-----------|----------|
| Fuerza hidrodinámica SPH | fx, fy, fz | N |
| Momento hidrodinámico SPH | mx, my, mz | N·m |
| Fuerza de contacto | cfx, cfy, cfz | N |
| Momento de contacto | cmx, cmy, cmz | N·m |

**Formato CSV:** Separador punto y coma (`;`), decimal punto (`.`), precisión 6 cifras significativas.

### 3.5.3 Criterio de Fallo (Movimiento Incipiente)

Se define movimiento incipiente cuando el bloque excede un umbral de desplazamiento relativo a su diámetro equivalente:

$$\frac{\Delta_{CM}}{d_{eq}} > \text{umbral}$$

donde Δ_CM es la distancia euclidiana del centro de masa respecto a su posición inicial.

Los umbrales específicos (% del diámetro equivalente para desplazamiento y grados para rotación) están pendientes de validación académica con el director de tesis.

**Nota sobre fuerza de contacto:** El estudio de convergencia (Sección 3.6) demostró que la fuerza de contacto Chrono no converge con el refinamiento de malla (CV = 82%), por lo que **no se utiliza como criterio de fallo**. Este hallazgo se discute en detalle en el Capítulo 4.

---

## 3.6 Estudio de Convergencia de Malla

### 3.6.1 Objetivo

Antes de la campaña paramétrica, se realizó un estudio de convergencia de malla para determinar la resolución espacial (dp) mínima que produce resultados independientes de la discretización.

### 3.6.2 Resoluciones Evaluadas

Se evaluaron 7 resoluciones espaciales, desde dp = 0.020 m (gruesa) hasta dp = 0.003 m (fina):

| dp (m) | Partículas | dim_min/dp | Tiempo GPU (min) |
|--------|------------|------------|------------------|
| 0.020 | 209,103 | 2.0 | 13.2 |
| 0.015 | 495,652 | 2.7 | 11.7 |
| 0.010 | 1,672,824 | 4.0 | 23.7 |
| 0.008 | 3,267,234 | 5.0 | 30.3 |
| 0.005 | 13,382,592 | 8.0 | 117.8 |
| 0.004 | 26,137,875 | 10.0 | 260.1 |
| 0.003 | 61,956,444 | 13.3 | 812.1 |

donde dim_min/dp es el número de partículas en la dimensión mínima del bloque (4.0 cm).

**Tiempo total de cómputo GPU:** 1,269 minutos (21.1 horas) en RTX 5090 (32 GB VRAM).

### 3.6.3 Criterio de Convergencia

Se emplea el criterio de cambio relativo (delta porcentual) entre resoluciones consecutivas:

$$\delta_{\%} = \frac{|f_{dp_i} - f_{dp_{i-1}}|}{|f_{dp_{i-1}}|} \times 100$$

Se considera convergida una métrica cuando δ% < 5% entre las dos resoluciones más finas.

### 3.6.4 Métricas Evaluadas

Se monitorearon cinco métricas independientes:

1. **Desplazamiento máximo del centro de masa** (métrica primaria)
2. **Fuerza hidrodinámica SPH máxima**
3. **Velocidad máxima del bloque**
4. **Rotación total acumulada**
5. **Fuerza de contacto máxima**

### 3.6.5 Resultados

| Métrica | δ% (dp=0.004 → 0.003) | Veredicto |
|---------|------------------------|-----------|
| Desplazamiento | 3.9% | **CONVERGIDO** |
| Fuerza SPH | 2.8% | **CONVERGIDA** |
| Velocidad | 0.8% | **CONVERGIDA** |
| Rotación | 6.3% | Estabilizada |
| Fuerza de contacto | CV = 82% | **NO CONVERGE** |

Las tres métricas primarias (desplazamiento, fuerza SPH, velocidad) presentan δ% < 5% entre las dos resoluciones más finas, confirmando convergencia.

**Hallazgo: No convergencia de la fuerza de contacto.** La fuerza de contacto entre el bloque y el fondo presenta un coeficiente de variación del 82% entre las 7 resoluciones, sin tendencia monótona. Este comportamiento es inherente a la naturaleza discreta del contacto en SPH, donde la geometría de las partículas en la interfaz cambia con cada resolución. Este hallazgo tiene implicancias para estudios de impacto en SPH y se documenta como contribución científica de esta tesis. Ver Figuras 3.1 a 3.9.

### 3.6.6 Resolución Seleccionada

Se selecciona **dp = 0.004 m** para la campaña paramétrica por las siguientes razones:

1. **Convergencia verificada:** δ < 5% en las tres métricas primarias respecto a dp = 0.003
2. **Resolución adecuada:** 10 partículas en la dimensión mínima del bloque (criterio mínimo)
3. **Eficiencia computacional:** 260 min por caso vs. 812 min para dp = 0.003 (3.1× más rápido)
4. **Viabilidad de producción:** 50 casos × 260 min = 217 horas (9 días) en RTX 5090

---

## 3.7 Pipeline Computacional

### 3.7.1 Arquitectura

El pipeline se implementa en Python y consta de 4 módulos desacoplados:

```
┌──────────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Geometry Builder │───►│ Batch Runner │───►│ Data Cleaner │───►│ ML Surrogate │
│  (STL + XML)      │    │ (GPU SPH)    │    │ (CSV → SQL)  │    │ (GP Regress) │
└──────────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

**Módulo 1 — Geometry Builder:** Toma la geometría STL del bloque y una plantilla XML base. Usando `trimesh`, calcula centro de masa, volumen, e inercia del bloque. Inyecta estos valores junto con los parámetros del experimento (altura de columna, masa, rotación) en el XML mediante `lxml`.

**Módulo 2 — Batch Runner:** Ejecuta la cadena GenCase → DualSPHysics para cada caso. Maneja timeouts, captura logs, y garantiza la limpieza de archivos binarios temporales (`.bi4`) en un bloque `try/finally`. Los archivos `.bi4` pueden generar hasta 17 GB por simulación.

**Módulo 3 — Data Cleaner (ETL):** Extrae cinemática del bloque desde `ChronoExchange_mkbound_51.csv`, fuerzas desde `ChronoBody_forces.csv`, y datos del flujo desde los archivos Gauge. Aplica criterios de fallo y almacena resultados en SQLite.

**Módulo 4 — ML Surrogate:** Entrena un regresor de procesos gaussianos (`GaussianProcessRegressor`, scikit-learn) sobre los resultados de la campaña paramétrica. Kernel Matérn (ν = 2.5) con validación cruzada Leave-One-Out.

### 3.7.2 Diseño de Experimentos

La campaña paramétrica emplea Muestreo por Hipercubo Latino (LHS) con `scipy.stats.qmc.LatinHypercube` (semilla = 42) para distribuir uniformemente los casos en el espacio de parámetros.

**Parámetros de entrada (pendientes de validación):**

| Parámetro | Rango tentativo | Unidad |
|-----------|----------------|--------|
| Altura columna de agua | 0.2 – 0.5 | m |
| Masa del bloque | 1.0 – 3.0 | kg |
| Ángulo de rotación Z | 0 – 90 | grados |

**Número de casos:** 50 (modo producción)

### 3.7.3 Hardware

| Componente | Desarrollo (laptop) | Producción (workstation) |
|------------|---------------------|--------------------------|
| GPU | RTX 4060 (8 GB VRAM) | RTX 5090 (32 GB VRAM) |
| CPU | i7-14650HX | — |
| Uso | Pruebas, dp ≥ 0.02 | Campaña completa, dp = 0.004 |

---

## 3.8 Figuras del Capítulo

| Figura | Archivo | Descripción |
|--------|---------|-------------|
| Fig. 3.1 | `fig01_desplazamiento_convergencia.png` | Convergencia del desplazamiento vs. dp |
| Fig. 3.2 | `fig02_fuerza_sph_convergencia.png` | Convergencia de la fuerza SPH vs. dp |
| Fig. 3.3 | `fig03_tasa_convergencia.png` | Tasa de cambio (δ%) por métrica |
| Fig. 3.4 | `fig04_fuerza_contacto_diagnostico.png` | Diagnóstico de no-convergencia de F_contacto |
| Fig. 3.5 | `fig05_velocidad_rotacion.png` | Velocidad y rotación del bloque |
| Fig. 3.6 | `fig06_costo_computacional.png` | Costo computacional vs. resolución |
| Fig. 3.7 | `fig07_tabla_resumen.png` | Tabla resumen de convergencia |
| Fig. 3.8 | `fig08_historia_completa.png` | Historia temporal completa (7 dp) |
| Fig. 3.9 | `fig09_veredicto.png` | Veredicto de convergencia |

Todas las figuras se encuentran en `data/figuras_7dp_es/pngs/` (versión en español).
