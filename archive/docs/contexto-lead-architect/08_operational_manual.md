# OPERATIONAL MANUAL & BUG FIX REGISTRY

## 1. ESPECIFICACIONES DEL ENTORNO (ENVIRONMENT)

Para garantizar la reproducibilidad entre la Estacion de Desarrollo (Laptop) y la Estacion de Produccion (Workstation), se deben respetar las siguientes rutas y versiones:

* **DualSPHysics Version:** v5.4.355 (08-04-2025).
* **GenCase Version:** v5.4.354.01 (07-04-2025).
* **Python Version:** 3.10 o superior.
* **Librerias Criticas:** `pandas`, `numpy`, `lxml`, `trimesh`, `scipy`, `sqlalchemy`.
* **Hardware Produccion:** NVIDIA RTX 5090 (32GB VRAM).
* **Hardware Desarrollo:** NVIDIA RTX 4060 (8GB VRAM).

---

## 2. HISTORIAL DE BUGS CRITICOS Y SOLUCIONES (DEBUGGING LOGS)

### BUG 01: Incompatibilidad de Formato de Inercia en XML

* **Sintoma:** GenCase crasheaba con el error: `*** Exception: Error reading xml - Attribute 'z' is missing`.
* **Causa Raiz:** El Modulo 1 generaba la inercia en formato de matriz 3x3 usando etiquetas `<values>` (estandar para el motor DEM, `RigidAlgorithm=2`). Sin embargo, el motor **ProjectChrono** (`RigidAlgorithm=3`) exige la diagonal del tensor en una sola etiqueta.
* **Solucion:** Se modifico `geometry_builder.py` para inyectar la inercia en el formato plano compatible con Chrono:
  ```xml
  <inertia x="Ixx_val" y="Iyy_val" z="Izz_val" />
  ```
* **Nota para el futuro:** Los terminos cruzados de inercia (Ixy, Ixz, Iyz) se desprecian, ya que Chrono alinea el solido con sus ejes principales de forma interna.

### BUG 02: Fallo de CWD (Current Working Directory) en GenCase

* **Sintoma:** GenCase no encontraba los archivos `.stl` referenciados en el XML, arrojando: `*** Exception: Cannot open the file. File: BLIR3.stl`.
* **Causa Raiz:** `subprocess.run()` se ejecutaba desde la raiz del proyecto, pero los XML de caso usan rutas relativas para los modelos 3D.
* **Solucion:** Se modifico el Modulo 2 (`batch_runner.py`) para que el proceso de GenCase se ejecute con el parametro `cwd=case_dir` (la subcarpeta donde vive el XML y sus STLs).

### BUG 03: Paths de Salida en el Solver GPU

* **Sintoma:** DualSPHysics iniciaba pero no podia escribir los archivos `.bi4` o leia un XML vacio.
* **Causa Raiz:** Windows maneja de forma inconsistente los separadores de ruta (`/` vs `\`) cuando se mezclan rutas absolutas y relativas en los flags del solver.
* **Solucion:** Se estandarizo el uso de `pathlib` de Python para generar rutas relativas estrictas desde el `case_dir` hacia la carpeta de salida `out/`.

---

## 3. PROTOCOLO DE GESTION DE DATOS BINARIOS (PURGE SYSTEM)

El sistema genera archivos de una densidad masiva (Big Data). Una simulacion a dp = 0.005m puede generar **15 GB de datos en 30 minutos**. Sin supervision, la RTX 5090 saturara el disco de 2TB en menos de un dia.

### Reglas del Basurero Algoritmico (Modulo 2):

El bloque `finally` en `run_case()` debe asegurar la eliminacion de los siguientes patrones de archivo tras cada corrida:

1. **`.bi4`**: Posicion y velocidad de todas las particulas del sistema. (Borrar sin piedad).
2. **`.cbi4`**: Geometria temporal de colision de Chrono. (Borrar).
3. **`.vtk / .vtp`**: Archivos de visualizacion para ParaView. (Borrar en produccion, conservar solo en desarrollo para debugging visual).
4. **Carpetas Temporales:** `/particles`, `/boundary`, `/surface`. (Eliminar contenido).

**Archivos que deben persistir (ORO):**

* Todos los `.csv` generados en la carpeta `data/` del caso.
* El archivo `Run.csv` (contiene el conteo de particulas y tiempo de GPU).
* El archivo `ChronoExchange_mkbound_51.csv` (cinematica del bloque).

---

## 4. LOGICA DE EXTRACCION ETL (MODULO 3)

### Procesamiento de Series de Tiempo (Pandas):

1. **Criterio de Falla de Diego (Auditado):** Antes era visual ("mirar el video").
2. **Criterio de Falla de Sebastian (Automatizado):**
   * Se extrae la posicion inicial del Centro de Masa (CM_0) en el instante t=0.5s (post-asentamiento).
   * Se calcula la distancia euclidiana en cada timestep.
   * Si Distancia > (d_eq x 0.05), se activa el flag `moved_flag = 1`.
   * Se integra la velocidad angular (omega) para obtener el angulo de rotacion neto. Si es > 5 grados, se activa el flag de movimiento.

### Manejo de la "Muerte Termica" de los Sensores:

* Los sensores de DualSPHysics arrojan el valor float minimo `-3.40282e+38` cuando no hay fluido presente.
* **Mandato:** Cualquier calculo de media o maximo debe usar `skipna=True` tras reemplazar estos valores por `NaN`, para evitar que las estadisticas de la ola se contaminen con valores negativos infinitos.

---

## 5. METRICAS DE RENDIMIENTO (BENCHMARKS)

Basado en las pruebas de febrero de 2026:

| Hardware | dp (m) | Particulas | Tiempo Simulado | Tiempo Real |
| :--- | :--- | :--- | :--- | :--- |
| RTX 4060 | 0.020 | 200 K | 10 s | 15.4 min |
| RTX 5090 | 0.020 | 200 K | 10 s | ~1.5 min |
| RTX 5090 | 0.005 | ~15 M | 10 s | ~3.5 horas |
| RTX 5090 | 0.003 | ~50 M | 10 s | ~24-36 horas |

---

## 6. GUIA DE TROUBLESHOOTING

Si el sistema falla en el futuro, revisa en este orden:

1. **VRAM Limit:** Si el error es `Cuda Error: Out of Memory`, reducir el dp o recortar el dominio de 15m a 12m.
2. **STL Non-Manifold:** Si GenCase falla al rellenar el solido, usar `trimesh.repair.fill_holes()` en el Modulo 1.
3. **Permissions Error:** En Windows, a veces el solver no puede borrar los `.bi4` porque el proceso de *MeasureTool* o *ParaView* aun los tiene abiertos. Implementar un `time.sleep(2)` antes de la purga.
