# Capítulo 5: Pipeline Computacional

## 5.1 Introducción

Para ejecutar una campaña paramétrica de 50+ simulaciones SPH de manera eficiente y reproducible, se desarrolló un pipeline computacional automatizado en Python. Este capítulo describe la arquitectura del pipeline, sus módulos, los mecanismos de robustez implementados, y las decisiones de diseño que garantizan la integridad de los resultados.

---

## 5.2 Arquitectura General

El pipeline consta de 5 módulos desacoplados conectados secuencialmente, orquestados por un módulo central:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         main_orchestrator.py                                  │
│                                                                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ geometry │─►│  batch   │─►│   data   │─►│    ml    │─►│    uq    │     │
│  │ builder  │  │  runner  │  │ cleaner  │  │surrogate │  │ (Sobol+  │     │
│  │(STL→XML) │  │(GPU SPH) │  │(CSV→SQL) │  │ (GP Reg) │  │  MC+CI)  │     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │
│                                                                               │
│  Entrada: experiment_matrix.csv (LHS)                                         │
│  Salida:  results.sqlite + gp_surrogate.pkl + sobol_indices + CI bands       │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Principio de diseño:** Cada módulo puede ejecutarse independientemente, facilitando el debug y la re-ejecución parcial en caso de fallo.

---

## 5.3 Módulo 1: Geometry Builder

**Archivo:** `src/geometry_builder.py`
**Función:** Genera un archivo XML de caso DualSPHysics a partir de una plantilla, una geometría STL, y un conjunto de parámetros.

### 5.3.1 Proceso

1. **Carga de STL:** Se lee el archivo STL del bloque (`BLIR3.stl`) usando `trimesh`
2. **Cálculo de propiedades físicas:** Volumen, centro de masa, tensor de inercia (3×3), diámetro equivalente
3. **Transformaciones geométricas:** Escalado, traslación y rotación del bloque según parámetros del caso
4. **Inyección XML:** Modificación de la plantilla base (`template_base.xml`) vía `lxml`:
   - `dp` (resolución espacial)
   - `<drawfilestl>` con `<drawscale>`, `<drawmove>`, `<drawrotate>`
   - `<fillbox>` con semilla en el centroide (solución al cuerpo hueco)
   - `<massbody>` con masa real en kg
   - `<inertia>` con valores de trimesh (corrige sobreestimación de GenCase)
   - `<FtPause>` = 0.5 s
   - Altura de columna de agua (dam_height)
5. **Copia de archivos auxiliares:** STLs y `Floating_Materials.xml` al directorio del caso

### 5.3.2 Correcciones Implementadas

| Problema | Solución |
|----------|----------|
| STL genera cuerpo hueco | `fillbox void` con semilla en centroide |
| GenCase sobreestima inercia (2-3×) | Inyección de inercia desde trimesh |
| `rhopbody` da masa incorrecta | Uso de `massbody` con masa real |
| Bloque salta al inicio | FtPause = 0.5 s obligatorio |

---

## 5.4 Módulo 2: Batch Runner

**Archivo:** `src/batch_runner.py`
**Función:** Ejecuta la cadena GenCase → DualSPHysics GPU para cada caso, con manejo de errores y limpieza garantizada.

### 5.4.1 Cadena de Ejecución

```
GenCase  CaseName_Def  outdir/CaseName  -save:all     (cwd = case_dir)
    ↓
DualSPHysics5.4_win64.exe  -gpu:0  outdir/CaseName  outdir
    ↓
[Chrono genera CSVs automáticamente durante simulación]
    ↓
Verificar CSVs → Recolectar a data/processed/ → Limpiar .bi4
```

### 5.4.2 Mecanismos de Robustez

**Timeout adaptativo por resolución:**

| dp (m) | Timeout (s) | Basado en |
|--------|-------------|-----------|
| 0.020 | 1,200 | 1.5× medido (13 min) |
| 0.010 | 2,400 | 1.5× medido (24 min) |
| 0.005 | 10,800 | 1.5× medido (118 min) |
| 0.004 | 28,800 | 1.5× medido (260 min) |
| 0.003 | 86,400 | 1.5× medido (812 min) |

El timeout se selecciona automáticamente según el dp del caso. Si el dp no está en la tabla, se usa el timeout del dp más cercano mayor.

**Limpieza blindada (try/finally):**
```python
try:
    # GenCase + DualSPHysics
finally:
    cleanup_binaries(case_dir, out_dir)  # SIEMPRE se ejecuta
```

Los archivos `.bi4` (hasta 17 GB por simulación) se eliminan garantizadamente incluso si la simulación falla o se excede el timeout. La función `cleanup_binaries` no lanza excepciones — loguea errores y continúa.

**Retry en limpieza de directorio:**
En Windows, `shutil.rmtree()` puede fallar si un proceso mantiene un lock en archivos. Se implementa retry con backoff exponencial (3 intentos).

**Verificación de outputs:**
Antes de declarar éxito, se verifica que `ChronoExchange_mkbound_*.csv` existe y tiene más de 100 bytes (no está vacío o truncado).

---

## 5.5 Módulo 3: Data Cleaner (ETL)

**Archivo:** `src/data_cleaner.py`
**Función:** Extrae datos de los CSVs de Chrono y Gauges, aplica criterios de fallo, y almacena resultados en SQLite.

### 5.5.1 Fuentes de Datos

| Archivo | Contenido | Separador |
|---------|-----------|-----------|
| ChronoExchange_mkbound_51.csv | Cinemática del bloque | `;` |
| ChronoBody_forces.csv | Fuerzas SPH + contacto | `;` |
| GaugesVel_V*.csv | Velocidad del flujo | `;` |
| GaugesMaxZ_hmax*.csv | Altura máxima del agua | `;` |

### 5.5.2 Procesamiento

1. **Lectura de ChronoExchange:** `pd.read_csv(path, sep=';')`, renombre de columnas con unidades
2. **Filtrado temporal:** Descarta datos antes de FtPause (t < 0.5 s) donde el bloque está congelado
3. **Cálculo de métricas:**
   - Desplazamiento: distancia euclidiana del centro de masa respecto a posición inicial
   - Rotación: integral acumulativa de la magnitud de velocidad angular × dt
   - Velocidad máxima: norma del vector velocidad
4. **Fuerzas:** Parse especial de ChronoBody_forces (headers duplicados por body)
5. **Gauges:** Reemplazo de sentinel (-3.40282e+38) por NaN, selección del gauge más cercano al bloque
6. **Criterio de fallo:** Desplazamiento > umbral × d_eq o rotación > umbral grados
7. **Persistencia:** SQLite con upsert (evita duplicados)

### 5.5.3 Tratamiento de Datos Anómalos

| Anomalía | Tratamiento |
|----------|-------------|
| Sentinel value (-3.40282e+38) | Reemplazo por NaN |
| Headers duplicados en forces | Parser manual con prefijo por body |
| CSV truncado (timeout) | Verificación de tamaño mínimo |

---

## 5.6 Módulo 4: ML Surrogate

**Archivo:** `src/ml_surrogate.py`
**Función:** Entrena un modelo de procesos gaussianos para predecir movimiento incipiente sin simular cada caso.

### 5.6.1 Configuración del Modelo

| Componente | Valor |
|------------|-------|
| Algoritmo | GaussianProcessRegressor (scikit-learn) |
| Kernel | Matérn (ν = 2.5) + WhiteKernel |
| Features | dam_height, boulder_mass, boulder_rot_z |
| Target | max_displacement |
| Preprocesamiento | StandardScaler |
| Validación | Leave-One-Out Cross-Validation |

### 5.6.2 Flujo de Entrenamiento

1. Carga resultados desde SQLite (`results.sqlite`)
2. Si n_real < 10: aumenta con datos sintéticos (flag "synthetic")
3. Normalización de features y target
4. Entrenamiento GP con optimización de hiperparámetros
5. Validación LOO con R² y error absoluto medio
6. Exportación: modelo pickle + figuras de validación

---

## 5.7 Orquestador de Producción

**Archivo:** `run_production.py`
**Función:** Script principal ("botón rojo") que coordina la campaña completa.

### 5.7.1 Modos de Ejecución

| Comando | Acción |
|---------|--------|
| `--generate 50` | Solo genera matriz LHS (50 muestras) |
| `--dry-run` | Simula campaña sin GPU |
| (sin flags) | Ejecuta en modo desarrollo (dp=0.02) |
| `--prod` | Ejecuta en modo producción (dp=0.004) |
| `--prod --desde 15` | Recovery: continúa desde caso 15 |

### 5.7.2 Mecanismos de Producción

**Pre-flight check:** Antes de iniciar, verifica:
- Ejecutables DualSPHysics existen
- Template XML y STL del boulder existen
- Espacio en disco disponible

**Monitoreo remoto:** Escribe `production_status.json` después de cada caso (escritura atómica vía archivo temporal), permitiendo monitorear progreso desde laptop.

**Crash safety:** Guarda resultados a SQLite después de cada caso exitoso. Si la máquina se reinicia, `--desde N` permite continuar sin re-ejecutar casos completados.

**Abort automático:** Si la tasa de fallos supera 30% (mínimo 3 casos ejecutados), el pipeline se aborta automáticamente. Esto evita desperdiciar horas de GPU en una configuración incorrecta.

**Re-entrenamiento automático:** Si hay ≥ 10 resultados exitosos al finalizar, re-entrena el modelo GP surrogate.

---

## 5.8 Diseño de Experimentos

### 5.8.1 Muestreo por Hipercubo Latino (LHS)

Se emplea LHS (`scipy.stats.qmc.LatinHypercube`, semilla = 42) para distribuir uniformemente los puntos en el espacio de parámetros, maximizando la cobertura con un número limitado de simulaciones.

**Ventajas sobre muestreo aleatorio:**
- Garantiza que cada estrato del espacio de parámetros es muestreado exactamente una vez
- Reduce la varianza del estimador para el mismo número de muestras
- Reproducible con semilla fija

### 5.8.2 Parámetros del Espacio

| Parámetro | Rango | Unidad | Estado |
|-----------|-------|--------|--------|
| Altura columna de agua | [0.10, 0.50] | m | Tentativo |
| Masa del bloque | [0.80, 1.60] | kg | Tentativo |
| Ángulo rotación Z | [0, 90] | ° | Tentativo |

**Número de casos:** 50 (producción), 5 (desarrollo)

---

## 5.9 Infraestructura de Hardware

| | Laptop (desarrollo) | Workstation (producción) |
|---|---|---|
| GPU | RTX 4060 (8 GB) | RTX 5090 (32 GB) |
| CPU | i7-14650HX | i9-14900KF |
| Uso | Pruebas dp ≥ 0.02 | Campaña dp = 0.004 |
| Tiempo por caso | ~13 min (dp=0.02) | ~260 min (dp=0.004) |
| Deploy | git push | Invoke-WebRequest ZIP |

**Estimación de producción:** 50 casos × 260 min = 217 horas ≈ 9 días continuos en RTX 5090.

---

## 5.10 Gestión de Almacenamiento

### 5.10.1 Presupuesto de Disco

| Componente | Tamaño | Persistencia |
|------------|--------|-------------|
| Partículas .bi4 (por caso) | ~17 GB | TEMPORAL (borrado post-ETL) |
| CSVs procesados (por caso) | ~50 MB | PERMANENTE |
| SQLite completo | ~10 MB | PERMANENTE |
| Total temporal máximo | ~17 GB | 1 caso a la vez |
| Total permanente (50 casos) | ~2.5 GB | Acumulativo |

### 5.10.2 Estrategia de Limpieza

Los archivos `.bi4` (partículas binarias) son el principal consumidor de disco. Se eliminan **inmediatamente después de verificar que los CSVs de salida existen**, dentro de un bloque `try/finally` que garantiza la limpieza incluso en caso de error.

---

## 5.11 Stack Tecnológico

| Componente | Tecnología | Versión |
|------------|------------|---------|
| Simulación SPH | DualSPHysics | v5.4.355 |
| Generación de geometría | GenCase | v5.4.354.01 |
| Dinámica de cuerpo rígido | ProjectChrono | (integrado en DualSPHysics) |
| Propiedades geométricas | trimesh | ≥ 3.0 |
| Manipulación XML | lxml | ≥ 4.0 |
| Análisis de datos | pandas | ≥ 1.5 |
| Diseño de experimentos | scipy (LHS) | ≥ 1.10 |
| Base de datos | SQLite | 3 (via sqlalchemy) |
| Modelo surrogate | scikit-learn (GP) | ≥ 1.3 |
| Visualización | matplotlib | ≥ 3.7 |
| Lenguaje | Python | 3.10+ |

---

## 5.12 Módulo 5: Cuantificación de Incertidumbre (UQ)

### 5.12.1 Motivación

Los resultados del GP surrogate son predicciones puntuales. Sin embargo, los parámetros de entrada tienen incertidumbre inherente: la masa de un boulder no se conoce con precisión exacta, la altura del tsunami varía, la fricción depende de la rugosidad local. Para transformar las predicciones en resultados con confianza estadística, se emplea Monte Carlo sobre el surrogate.

### 5.12.2 Monte Carlo sobre el GP

1. Se definen distribuciones de probabilidad para cada parámetro de entrada (ej. masa ~ Uniform(0.80, 1.60), altura ~ Uniform(0.10, 0.50))
2. Se generan 10,000 muestras aleatorias del espacio de parámetros
3. Cada muestra se evalúa en el GP surrogate (milisegundos por evaluación)
4. Se construyen distribuciones de los outputs: desplazamiento, rotación, fuerza
5. Se calculan intervalos de confianza al 95% y probabilidades de movimiento incipiente

**Justificación computacional:** Monte Carlo directo sobre DualSPHysics requeriría 10,000 × 4 horas = 4.5 años de cómputo. Sobre el GP surrogate, las 10,000 evaluaciones toman ~10 segundos.

### 5.12.3 Análisis de Sensibilidad Global (Índices de Sobol)

Los índices de Sobol (Sobol', 2001) descomponen la varianza total de una salida en contribuciones de cada parámetro:

- **Índice de primer orden (S_i):** fracción de varianza explicada por el parámetro i actuando solo
- **Índice de orden total (ST_i):** fracción de varianza explicada por el parámetro i incluyendo interacciones con otros

Se calculan mediante el esquema de muestreo de Saltelli (2002) sobre el GP surrogate. Esto identifica qué parámetro domina la incertidumbre del resultado, guiando tanto la interpretación física como futuras campañas experimentales.

### 5.12.4 Precedente Metodológico

Este pipeline (simulación SPH → GP emulador → Monte Carlo → Sobol) sigue el enfoque de Salmanidou et al. (2017, 2020), quienes lo aplicaron a cuantificación de incertidumbre en oleaje por deslizamiento de tierra. La diferencia es la aplicación al transporte de bloques costeros con geometría irregular.

### Referencias UQ

- Oakley, J.E. & O'Hagan, A. (2004). *JRSS-B*, 66(3), 751-769. DOI: 10.1111/j.1467-9868.2004.05304.x
- Sobol', I.M. (2001). *Math. Comput. Simul.*, 55(1-3), 271-280. DOI: 10.1016/S0378-4754(00)00270-6
- Saltelli, A. (2002). *Comput. Phys. Commun.*, 145(2), 280-297. DOI: 10.1016/S0010-4655(02)00280-1
- Salmanidou, D.M., et al. (2020). *Water*, 12(2), 416. DOI: 10.3390/w12020416
