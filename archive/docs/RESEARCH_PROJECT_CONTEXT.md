# CONTEXTO TÉCNICO DE INVESTIGACIÓN: Proyecto SPH-IncipientMotion

## 1. DEFINICIÓN DEL PROBLEMA CIENTÍFICO

**Tema:** Determinación de umbrales de movimiento incipiente en bloques costeros (boulders) sometidos a flujos de alta energía (tipo Tsunami).
**Fenómeno Físico:** Interacción Fluido-Estructura (FSI) turbulenta con superficie libre.
**Variable Crítica:** El instante exacto ($t_{crit}$) en que las fuerzas hidrodinámicas (Arrastre + Sustentación) superan las fuerzas estabilizadoras (Peso + Fricción), rompiendo el equilibrio estático.

## 2. OBJETIVOS DE LA INVESTIGACIÓN

1. **Determinar la "Ley de Falla":** Encontrar la relación matemática entre la velocidad del flujo ($u$), la altura de ola ($h$), la masa del bloque ($M$) y su forma geométrica.
2. **Validación Numérica:** Utilizar el método SPH para replicar condiciones físicas complejas que las fórmulas empíricas (Nandasena, Engel & May) simplifican en exceso.
3. **Generación de Modelo Predictivo:** Crear un *Surrogate Model* mediante Machine Learning que prediga la estabilidad de un bloque sin necesidad de simular todos los escenarios posibles.

## 3. METODOLOGÍA COMPUTACIONAL (EL MOTOR)

### A. Software Core: DualSPHysics (v5.2+)

* **Método:** Smoothed Particle Hydrodynamics (Lagrangiano, sin malla).
* **Justificación:** Capacidad superior para modelar grandes deformaciones de la superficie libre, salpicaduras y fuerzas de impacto violentas sobre sólidos flotantes/móviles.
* **Parámetros Clave a Calibrar:**
  * `Dp` (Distancia entre partículas): Define la resolución y precisión.
  * `Coeficiente de Fricción` (Suelo-Bloque).
  * `Viscosidad Artificial`: Para estabilidad numérica.
  * `CFL Number`: Para estabilidad temporal.

### B. Infraestructura de Hardware

* **Entorno de Desarrollo (Local):** Laptop i7-14650HX + RTX 4060.
  * *Uso:* Generación de geometría, scripts de pre-proceso, pruebas de concepto (baja resolución), limpieza de datos.
* **Entorno de Producción (High Performance Computing):** Workstation NVIDIA RTX 5090 (32GB VRAM).
  * *Uso:* Ejecución masiva de simulaciones de alta fidelidad (20M+ partículas) en modo Batch.

## 4. DESAFÍOS TÉCNICOS HEREDADOS (ESTADO ACTUAL)

La investigación previa (fase exploratoria de Diego) detectó obstáculos críticos que este proyecto debe resolver mediante automatización:

1. **Geometría "Hollow Shell":**
   * *Problema:* Al importar STLs irregulares, DualSPHysics crea una cáscara vacía sin masa interna correcta.
   * *Solución Requerida:* Algoritmo en Python que calcule volumen/centro de masa y genere automáticamente la etiqueta `<fillbox>` o `<fillpoint>` con las coordenadas transformadas según la rotación del bloque.
2. **Gestión de Datos (Input):**
   * *Problema:* Edición manual de archivos XML (propenso a error humano).
   * *Solución Requerida:* Script `xml_factory.py` que genere cientos de variaciones de casos (masa/ángulo) de forma procedural.
3. **Gestión de Datos (Output):**
   * **Problema: Archivos CSV gigantes, formatos inconsistentes (decimales punto/coma), análisis visual subjetivo.**
   * **Solución Requerida: Pipeline ETL (extract_clean_load.py) que parsee salidas de MeasureTool, aplique un criterio cinemático riguroso de falla (ej: Desplazamiento del Centro de Masa**

     <pre _ngcontent-ng-c951868740=""><strong _ngcontent-ng-c1482782766="" class="ng-star-inserted"><code _ngcontent-ng-c951868740="" class="rendered"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi mathvariant="normal">Δ</mi><mi>X</mi><mo separator="true">,</mo><mi mathvariant="normal">Δ</mi><mi>Y</mi><mo>></mo><mn>5</mn><mi mathvariant="normal">%</mi></mrow></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut"></span><span class="mord">Δ</span><span class="mord mathnormal">X</span><span class="mpunct">,</span><span class="mspace"></span><span class="mord">Δ</span><span class="mord mathnormal">Y</span><span class="mspace"></span><span class="mrel">></span><span class="mspace"></span></span><span class="base"><span class="strut"></span><span class="mord">5%</span></span></span></span></code></strong></pre>

     **** del diámetro equivalente, o rotación neta)** y guarde el resultado booleano en SQL.**

## 5. ESTRATEGIA DE EXPERIMENTACIÓN (PIPELINE)

El flujo de trabajo no es lineal manual, es cíclico y automatizado:

1. **Diseño de Experimentos (LHS):** Python genera una lista de 100-500 casos distribuidos inteligentemente en el espacio de variables.
2. **Generación:** Se crean 500 archivos XML con geometría corregida automáticamente.
3. **Simulación (Batch):** La RTX 5090 ejecuta una cola de trabajos secuenciales.
4. **Post-Proceso (Headless):** Al terminar cada simulación, se extraen métricas clave y se borran los binarios pesados (RAW data) conservando solo la "Inteligencia" (Processed data).
5. **Aprendizaje (ML):** Los datos alimentan un modelo de Procesos Gaussianos para refinar la búsqueda de umbrales.

## 6. ENTREGABLES TÉCNICOS ESPERADOS

* Repositorio de Código (Python scripts para automatización SPH).
* Base de Datos SQL con resultados paramétricos.
* Curvas de Estabilidad (Gráficos de Fase) generados por el modelo.
