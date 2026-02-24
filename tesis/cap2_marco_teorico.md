# Capítulo 2: Marco Teórico

## 2.1 Introducción

Este capítulo presenta los fundamentos teóricos que sustentan la investigación. Se abordan cuatro pilares: (1) la física del transporte de bloques costeros por tsunami y las ecuaciones predictivas existentes, (2) el método de Hidrodinámica de Partículas Suavizadas (SPH), (3) el acoplamiento con motores de cuerpos rígidos para la interacción fluido-estructura, y (4) la modelación surrogate mediante procesos gaussianos y análisis de sensibilidad global.

---

## 2.2 Transporte de Bloques Costeros por Tsunami

### 2.2.1 Depósitos de Bloques Costeros

Los depósitos de bloques costeros (*coastal boulder deposits*, CBD) son acumulaciones de grandes fragmentos de roca encontrados tierra adentro de la línea de costa, cuyo origen se atribuye a eventos de alta energía como tsunamis y tormentas extremas. Estos depósitos constituyen evidencia geológica de eventos pasados y permiten estimar la magnitud de tsunamis prehistóricos cuando no se dispone de registros instrumentales (Goto et al., 2010; Etienne et al., 2011).

La cadena causal que produce un CBD es:

1. Un terremoto submarino genera un tsunami.
2. El tsunami se propaga hacia la costa y se amplifica por efecto batimétrico.
3. El flujo impacta la zona costera y ejerce fuerzas hidrodinámicas sobre bloques de roca.
4. Si las fuerzas superan la resistencia del bloque (peso + fricción + inercia), se produce el transporte.
5. El bloque se deposita cuando la energía del flujo ya no es suficiente para sostener el movimiento.

El estudio de los CBD tiene aplicaciones directas en la evaluación de riesgo tsunámico, ya que permite acotar las velocidades y alturas de flujo de eventos históricos a partir de la masa y distancia de transporte de los bloques depositados.

### 2.2.2 Mecanismos de Transporte

Según Nott (2003), un bloque costero puede experimentar tres modos principales de transporte:

1. **Deslizamiento (*sliding*):** El bloque se desplaza horizontalmente cuando la fuerza de arrastre supera la fricción estática. El bloque mantiene contacto continuo con el sustrato.

2. **Volcamiento (*overturning/toppling*):** El bloque rota alrededor de una arista inferior cuando el momento hidrodinámico supera el momento estabilizador del peso. Este modo es particularmente relevante para bloques con relaciones de aspecto altas.

3. **Sustentación (*lifting/saltation*):** El bloque se eleva verticalmente cuando la fuerza de sustentación supera el peso sumergido. Este modo requiere velocidades de flujo elevadas y es más probable en bloques con geometrías que generan diferencias de presión significativas entre la cara superior e inferior.

En la práctica, el transporte real involucra combinaciones de estos modos. Un bloque puede iniciar con un ligero deslizamiento, luego volcarse parcialmente y finalmente ser arrastrado por el flujo en una trayectoria compleja que incluye rotaciones en múltiples ejes.

### 2.2.3 Ecuaciones Predictivas Clásicas

#### Nott (2003)

Nott propuso las primeras ecuaciones cuantitativas para estimar la velocidad mínima de flujo necesaria para transportar bloques costeros. Para un bloque subaerial (no sumergido previamente), la velocidad crítica de volcamiento es:

$$u^2 \geq \frac{2(\rho_s/\rho_w - 1) g a (b/c)}{C_D (b/a) + C_L}$$

donde $\rho_s$ es la densidad del bloque, $\rho_w$ la densidad del agua, $g$ la gravedad, y $a$, $b$, $c$ son las dimensiones del bloque (largo, ancho, alto). $C_D$ y $C_L$ son los coeficientes de arrastre y sustentación, respectivamente.

**Limitación principal:** Los bloques se modelan como prismas rectangulares con dimensiones $a \times b \times c$, y los coeficientes hidrodinámicos se asumen constantes ($C_D = 1.5$, $C_L = 0.178$) independientemente de la geometría real.

#### Nandasena et al. (2011)

Nandasena et al. refinaron las ecuaciones de Nott considerando explícitamente los tres modos de transporte (deslizamiento, volcamiento, sustentación) como criterios separados. El modo que requiere menor velocidad determina el mecanismo de fallo:

**Deslizamiento:**
$$u^2 \geq \frac{2(\rho_s/\rho_w - 1) g c (\mu_s \cos\theta - \sin\theta)}{C_D (c/b) + \mu_s C_L}$$

**Volcamiento:**
$$u^2 \geq \frac{2(\rho_s/\rho_w - 1) g c (c/b \cos\theta - \sin\theta)}{C_D (c^2/b^2) + C_L}$$

donde $\mu_s$ es el coeficiente de fricción estática y $\theta$ la pendiente del sustrato.

**Avance respecto a Nott:** Identifica el modo de fallo más probable para cada configuración geométrica. **Limitación persistente:** Sigue empleando geometría rectangular y coeficientes constantes.

#### Engel & May (2012)

Engel & May incorporaron la fracción de sumergimiento del bloque y la variación temporal del flujo, proporcionando estimaciones más realistas para bloques parcialmente expuestos. Sin embargo, mantienen la aproximación de prisma rectangular.

#### Síntesis de Limitaciones

Todas las ecuaciones predictivas comparten tres limitaciones fundamentales:

| Limitación | Implicancia |
|------------|-------------|
| Geometría rectangular | No captura irregularidades que modifican fuerzas y momentos |
| Coeficientes constantes | $C_D$ y $C_L$ dependen fuertemente de la geometría real |
| Equilibrio estático | Ignora la dinámica transitoria del impacto (aceleración del flujo, respuesta del bloque) |
| Modos de fallo aislados | En realidad los modos se combinan (traslación + rotación simultáneas) |

La presente tesis aborda estas limitaciones mediante simulación numérica directa, que resuelve la interacción fluido-estructura sin recurrir a coeficientes empíricos ni simplificaciones geométricas.

---

## 2.3 Hidrodinámica de Partículas Suavizadas (SPH)

### 2.3.1 Fundamentos del Método

SPH es un método numérico lagrangiano sin malla (*meshfree*) para resolver ecuaciones diferenciales parciales, originalmente desarrollado para problemas de astrofísica (Gingold & Monaghan, 1977; Lucy, 1977) y posteriormente adaptado a mecánica de fluidos (Monaghan, 1994).

La idea central de SPH es representar un campo continuo $A(\mathbf{r})$ como una suma ponderada sobre un conjunto de partículas vecinas:

$$A(\mathbf{r}) \approx \sum_j \frac{m_j}{\rho_j} A_j \, W(\mathbf{r} - \mathbf{r}_j, h)$$

donde $m_j$ y $\rho_j$ son la masa y densidad de la partícula $j$, $A_j$ es el valor del campo en la partícula $j$, y $W$ es la **función kernel** de suavizado con longitud característica $h$ (*smoothing length*).

La función kernel define el radio de influencia de cada partícula. Las propiedades requeridas son:
- Normalización: $\int W(\mathbf{r}, h) \, d\mathbf{r} = 1$
- Compacidad: $W(\mathbf{r}, h) = 0$ para $|\mathbf{r}| > \kappa h$ (soporte compacto)
- Convergencia al delta de Dirac: $\lim_{h \to 0} W(\mathbf{r}, h) = \delta(\mathbf{r})$

### 2.3.2 Ecuaciones Gobernantes

SPH resuelve las ecuaciones de Navier-Stokes para un fluido ligeramente compresible (*Weakly Compressible SPH*, WCSPH):

**Conservación de masa (ecuación de continuidad):**
$$\frac{d\rho_i}{dt} = \sum_j m_j (\mathbf{v}_i - \mathbf{v}_j) \cdot \nabla_i W_{ij}$$

**Conservación de momentum:**
$$\frac{d\mathbf{v}_i}{dt} = -\sum_j m_j \left(\frac{P_i}{\rho_i^2} + \frac{P_j}{\rho_j^2} + \Pi_{ij}\right) \nabla_i W_{ij} + \mathbf{g}$$

donde $P$ es la presión, $\Pi_{ij}$ es el término de viscosidad artificial, y $\mathbf{g}$ la aceleración gravitacional.

**Ecuación de estado (Tait):**
$$P = B \left[\left(\frac{\rho}{\rho_0}\right)^\gamma - 1\right]$$

con $\gamma = 7$ y $B = c_0^2 \rho_0 / \gamma$, donde $c_0$ es la velocidad del sonido numérica (típicamente $c_0 = 10 \, v_{max}$ para mantener variaciones de densidad < 1%).

### 2.3.3 Funciones Kernel

Las funciones kernel más empleadas en SPH son:

**Kernel cúbico (Monaghan & Lattanzio, 1985):**
Soporte compacto en $2h$. Ampliamente utilizado pero puede presentar inestabilidad tensil (*tensile instability*) en algunas configuraciones.

**Kernel Wendland C2 (Wendland, 1995):**
$$W(q) = \alpha_D (1 - q/2)^4 (2q + 1), \quad q = r/h \leq 2$$

donde $\alpha_D$ es la constante de normalización dimensional. El kernel Wendland es preferido para problemas con cuerpos flotantes por su mayor estabilidad y suavidad (Dehnen & Aly, 2012).

### 2.3.4 Viscosidad Artificial

Para estabilizar la simulación y prevenir oscilaciones no físicas en el campo de presión, se emplea viscosidad artificial (Monaghan, 1992):

$$\Pi_{ij} = \begin{cases} \frac{-\alpha \bar{c}_{ij} \mu_{ij}}{\bar{\rho}_{ij}} & \text{si } \mathbf{v}_{ij} \cdot \mathbf{r}_{ij} < 0 \\ 0 & \text{en caso contrario} \end{cases}$$

donde $\alpha$ es el coeficiente de viscosidad artificial. Valores típicos oscilan entre 0.01 y 0.1. Valores altos estabilizan la simulación pero introducen disipación numérica excesiva.

### 2.3.5 Difusión de Densidad (Delta-SPH)

El esquema WCSPH estándar produce oscilaciones de alta frecuencia en el campo de densidad (y por ende de presión). Para mitigar esto se introduce un término de difusión de densidad:

- **Molteni & Colagrossi (2009):** Primer término difusivo para SPH.
- **Fourtakas et al. (2019):** Delta-SPH generalizado (DDT tipo 2), compatible con condiciones de contorno y cuerpos flotantes. Es el esquema recomendado para simulaciones con acoplamiento fluido-estructura.

### 2.3.6 Integración Temporal

**Esquema Symplectic (Leapfrog):**
Integrador de segundo orden en tiempo que conserva las propiedades simplécticas del sistema hamiltoniano. Preferido sobre el esquema de Verlet por su mayor precisión y estabilidad a largo plazo.

**Condición CFL:**
El paso de tiempo se calcula dinámicamente según la condición de Courant-Friedrichs-Lewy:

$$\Delta t \leq C_{CFL} \cdot \frac{h}{c_0 + v_{max}}$$

con $C_{CFL}$ entre 0.1 y 0.4 (típicamente 0.2 para simulaciones con cuerpos rígidos).

### 2.3.7 Ventajas y Limitaciones de SPH

**Ventajas para el presente problema:**

| Ventaja | Relevancia |
|---------|------------|
| Sin malla | Maneja ruptura de superficie libre, salpicaduras y fragmentación |
| Lagrangiano | Sigue naturalmente el fluido — ideal para impactos transitorios |
| Geometrías complejas | Bloques irregulares se discretizan directamente desde STL |
| Acoplamiento FSI natural | Fuerzas fluido→sólido emergen de la formulación, sin interpolación |
| Paralelizable en GPU | Millones de partículas en horas (vs. días en CPU) |

**Limitaciones:**

| Limitación | Mitigación |
|------------|------------|
| Costo computacional alto | GPU + modelo surrogate |
| Condiciones de contorno complejas | Dynamic Boundary Conditions (DBC) |
| Ruido en campo de presión | Delta-SPH (Fourtakas et al., 2019) |
| Dependencia de resolución | Estudio de convergencia de malla obligatorio |

---

## 2.4 DualSPHysics

### 2.4.1 Descripción General

DualSPHysics es un código SPH de código abierto desarrollado por un consorcio de universidades europeas (Universidade de Vigo, University of Manchester, Università di Catania, entre otras). Su característica distintiva es la capacidad de ejecutar en CPU y GPU (NVIDIA CUDA), logrando aceleraciones de 100× o más en GPU respecto a CPU para problemas de millones de partículas (Crespo et al., 2015; Domínguez et al., 2022).

### 2.4.2 Cadena de Ejecución

El flujo de trabajo de DualSPHysics se compone de tres etapas:

1. **Pre-procesamiento (GenCase):** Lee una definición XML del caso y genera la distribución inicial de partículas en formato binario (.bi4). Interpreta geometrías primitivas (cajas, cilindros, esferas) y archivos STL.

2. **Simulación (DualSPHysics):** Resuelve las ecuaciones SPH en GPU, avanzando en tiempo hasta $t_{max}$. Genera instantáneas periódicas de las posiciones de todas las partículas.

3. **Post-procesamiento:** Herramientas auxiliares extraen campos de velocidad, presión, fuerzas sobre cuerpos, y exportan a formatos de visualización (VTK, CSV).

### 2.4.3 Condiciones de Contorno

DualSPHysics implementa partículas de contorno dinámicas (*Dynamic Boundary Conditions*, DBC) como condición de contorno por defecto (Crespo et al., 2007). Las partículas de contorno satisfacen las mismas ecuaciones que las partículas de fluido pero permanecen fijas (o se mueven según una ley prescrita). Cuando una partícula de fluido se acerca al contorno, la densidad de las partículas de contorno aumenta, generando una fuerza repulsiva que impide la penetración.

### 2.4.4 Precisión Posicional

Para dominios computacionales grandes ($L > 10$ m), la aritmética de precisión simple (32 bits) introduce errores de redondeo significativos en las posiciones de las partículas. DualSPHysics ofrece el modo `posdouble` que emplea precisión doble (64 bits) para las posiciones manteniendo precisión simple para el resto de las variables, con un incremento de costo computacional menor al 10%.

---

## 2.5 Acoplamiento con Cuerpos Rígidos (ProjectChrono)

### 2.5.1 Necesidad del Acoplamiento

En SPH estándar, los cuerpos flotantes se tratan como conjuntos de partículas de contorno que se mueven como sólido rígido bajo las fuerzas SPH resultantes. Sin embargo, este enfoque básico no resuelve colisiones entre cuerpos ni con el sustrato, lo cual es esencial para simular un bloque que reposa sobre una playa y debe resistir el impacto del flujo mediante fricción y contacto.

### 2.5.2 ProjectChrono

ProjectChrono es un motor de simulación multi-física de código abierto especializado en dinámica de cuerpos rígidos, detección de colisiones y resolución de restricciones (Tasora et al., 2016). DualSPHysics integra ProjectChrono como su solver de cuerpos rígidos cuando se activa `RigidAlgorithm = 3`.

El acoplamiento bidireccional funciona así:

1. **SPH → Chrono:** Las fuerzas y momentos hidrodinámicos calculados por SPH se transfieren a Chrono como cargas externas sobre el cuerpo rígido.
2. **Chrono → SPH:** Chrono resuelve la dinámica del cuerpo (incluyendo colisiones y fricción) y devuelve las nuevas posiciones y velocidades, que SPH usa para mover las partículas de contorno del cuerpo.

### 2.5.3 Resolución de Contactos

Chrono emplea el método NSC (*Non-Smooth Contacts*) para resolver contactos entre cuerpos. A diferencia de los métodos de penalización que introducen fuerzas proporcionales a la penetración, NSC resuelve las restricciones de contacto como un problema complementario lineal (LCP), evitando penetraciones y proporcionando fuerzas de contacto exactas en el sentido de Coulomb.

**Parámetros de material** relevantes para el contacto:
- **Módulo de Young:** rigidez del material
- **Razón de Poisson:** compresibilidad lateral
- **Coeficiente de restitución:** elasticidad del rebote (0 = perfectamente plástico, 1 = perfectamente elástico)
- **Coeficiente de fricción cinética:** resistencia al deslizamiento

### 2.5.4 Datos de Salida del Acoplamiento

El acoplamiento SPH-Chrono genera automáticamente durante la simulación:

- **ChronoExchange:** Cinemática completa del cuerpo rígido (posición, velocidad lineal, velocidad angular, aceleraciones) con resolución temporal de 0.001 s.
- **ChronoBody_forces:** Fuerzas SPH (incluyendo gravedad) y fuerzas de contacto Chrono por separado, permitiendo discriminar la contribución hidrodinámica de la contribución de contacto.

Esta separación es valiosa porque permite identificar el instante exacto del movimiento incipiente: cuando las fuerzas SPH superan la resistencia combinada de peso + fricción, el bloque comienza a moverse y las fuerzas de contacto se modifican.

---

## 2.6 Modelación Surrogate: Procesos Gaussianos

### 2.6.1 Motivación

Una campaña paramétrica completa — variando altura de flujo, masa del bloque, orientación y forma — requiere cientos o miles de simulaciones. A 4 horas por simulación en GPU, esto resulta prohibitivo. Un modelo surrogate (emulador) entrenado con un subconjunto de simulaciones permite:

1. Predecir resultados para combinaciones de parámetros no simuladas
2. Cuantificar la incertidumbre de las predicciones
3. Realizar análisis de sensibilidad con costo computacional negligible

### 2.6.2 Regresión de Procesos Gaussianos (GPR)

Un proceso gaussiano (GP) es una distribución sobre funciones, donde cualquier conjunto finito de evaluaciones sigue una distribución normal multivariada (Rasmussen & Williams, 2006). Formalmente:

$$f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))$$

donde $m(\mathbf{x})$ es la función de media (típicamente cero) y $k(\mathbf{x}, \mathbf{x}')$ es la función de covarianza (kernel).

Dado un conjunto de observaciones $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$, la predicción del GP para un nuevo punto $\mathbf{x}_*$ es:

$$\mu_* = \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{y}$$
$$\sigma_*^2 = k_{**} - \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{k}_*$$

donde $\mathbf{K}$ es la matriz de covarianza entre puntos de entrenamiento, $\mathbf{k}_*$ es el vector de covarianzas entre el punto nuevo y los puntos de entrenamiento, y $\sigma_n^2$ es la varianza del ruido.

**Ventaja clave:** El GP proporciona no solo una predicción puntual ($\mu_*$) sino también una medida de incertidumbre ($\sigma_*$), que crece naturalmente en regiones del espacio de parámetros con pocas observaciones.

### 2.6.3 Función Kernel

La elección del kernel codifica las suposiciones sobre la suavidad de la función a emular. Se emplea el kernel Matérn con $\nu = 5/2$:

$$k_{\text{Matérn}}(r) = \sigma_f^2 \left(1 + \frac{\sqrt{5} \, r}{\ell} + \frac{5 r^2}{3 \ell^2}\right) \exp\left(-\frac{\sqrt{5} \, r}{\ell}\right)$$

donde $r = |\mathbf{x} - \mathbf{x}'|$, $\ell$ es la longitud de escala, y $\sigma_f^2$ la varianza de la señal. Matérn 5/2 produce funciones dos veces diferenciables, lo cual es apropiado para respuestas físicas suaves pero no analíticas.

### 2.6.4 Validación

La validación del GP se realiza mediante Leave-One-Out Cross-Validation (LOO-CV), donde cada observación se predice usando un modelo entrenado con las $n-1$ restantes. Las métricas de validación son:

- **R² (coeficiente de determinación):** fracción de varianza explicada. R² > 0.9 indica un ajuste adecuado.
- **MAE (error absoluto medio):** magnitud típica del error de predicción en unidades físicas.
- **Cobertura del intervalo de confianza:** fracción de observaciones que caen dentro del intervalo de predicción al 95%.

---

## 2.7 Análisis de Sensibilidad Global

### 2.7.1 Motivación

El análisis de sensibilidad global identifica qué parámetros de entrada contribuyen más a la variabilidad de la respuesta. Esto es esencial para:
- Priorizar parámetros en futuras campañas experimentales
- Simplificar el modelo (fijar parámetros poco influyentes)
- Interpretar físicamente los resultados

### 2.7.2 Índices de Sobol

Los índices de Sobol (Sobol', 2001) descomponen la varianza total de la respuesta $Y = f(X_1, X_2, \ldots, X_d)$ en contribuciones de cada parámetro y sus interacciones:

$$V(Y) = \sum_i V_i + \sum_{i<j} V_{ij} + \cdots + V_{1,2,\ldots,d}$$

**Índice de primer orden ($S_i$):**
$$S_i = \frac{V_i}{V(Y)} = \frac{V[E(Y|X_i)]}{V(Y)}$$

Mide la fracción de varianza explicada por $X_i$ actuando solo, sin interacciones.

**Índice de orden total ($S_{T_i}$):**
$$S_{T_i} = 1 - \frac{V[E(Y|X_{\sim i})]}{V(Y)}$$

Mide la fracción de varianza explicada por $X_i$ incluyendo todas sus interacciones con otros parámetros. Si $S_{T_i} \approx 0$, el parámetro $X_i$ puede fijarse sin pérdida de información.

### 2.7.3 Cálculo mediante Saltelli (2002)

El esquema de muestreo de Saltelli permite estimar los índices de Sobol de primer orden y de orden total con $n(2d + 2)$ evaluaciones del modelo, donde $n$ es el tamaño base de la muestra y $d$ el número de parámetros. Al evaluar sobre el GP surrogate (milisegundos por evaluación), se pueden emplear muestras grandes ($n = 10,000$) para obtener estimaciones precisas.

### 2.7.4 Monte Carlo sobre el Surrogate

El método de Monte Carlo consiste en:

1. Definir distribuciones de probabilidad para los parámetros de entrada (ej. uniforme en los rangos del LHS)
2. Generar $N$ muestras aleatorias ($N = 10,000$)
3. Evaluar el GP para cada muestra
4. Construir la distribución empírica de la respuesta
5. Calcular estadísticas: media, varianza, percentiles, intervalos de confianza al 95%

**Eficiencia:** Monte Carlo directo sobre DualSPHysics requeriría $10,000 \times 4$ horas = 4.5 años. Sobre el GP surrogate, las 10,000 evaluaciones se completan en segundos.

---

## 2.8 Precedentes Metodológicos

### 2.8.1 SPH para Transporte de Bloques

El uso de SPH para simular el transporte de bloques costeros es relativamente reciente. Entre los trabajos más relevantes:

- **Imamura et al. (2008):** Simulación 2D de transporte de bloques por tsunami usando SPH, demostrando la viabilidad del enfoque.
- **Oetjen et al. (2020):** Simulación 3D de transporte de bloques con DualSPHysics, comparando resultados numéricos con experimentos de laboratorio para bloques cúbicos.
- **Nandasena & Tanaka (2013):** Revisión de las capacidades de los métodos numéricos para estudiar el transporte de bloques costeros.

### 2.8.2 Surrogate + UQ para Simulaciones de Tsunami

El pipeline de simulación → surrogate → Monte Carlo → Sobol que se emplea en esta tesis sigue el enfoque establecido por:

- **Salmanidou et al. (2017):** GP emulador para simulaciones de tsunami por deslizamiento de tierra, con análisis de sensibilidad de Sobol.
- **Salmanidou et al. (2020):** Extensión del enfoque GP + Monte Carlo a la cuantificación de incertidumbre en oleaje por deslizamiento, publicado en *Water*.
- **Sarri et al. (2012):** GP emulador para simulaciones de inundación por tsunami, demostrando que $\sim$50 simulaciones son suficientes para entrenar un emulador confiable en espacios de 3-4 dimensiones.

La diferencia de la presente tesis respecto a estos precedentes es la aplicación al **transporte de bloques con geometría irregular**, un problema donde la forma del cuerpo es una variable clave que las ecuaciones analíticas no capturan.

---

## 2.9 Síntesis del Marco Teórico

La Tabla 2.1 resume cómo cada componente teórica se integra en la investigación.

**Tabla 2.1: Integración de componentes teóricas en la investigación**

| Componente Teórica | Rol en la Tesis | Capítulo de Aplicación |
|--------------------|-----------------|-----------------------|
| Transporte de bloques | Define el problema físico y los modos de fallo | Cap. 1, 3 |
| Ecuaciones predictivas (Nott, Nandasena) | Referencia contra la cual se evalúa la mejora numérica | Cap. 6, 7 |
| SPH (WCSPH) | Método numérico para resolver la interacción fluido-bloque | Cap. 3, 4 |
| DualSPHysics + Chrono | Implementación del solver SPH + dinámica de cuerpos rígidos | Cap. 3, 5 |
| Procesos Gaussianos | Modelo surrogate para predicción rápida + incertidumbre | Cap. 5, 6 |
| Sobol + Monte Carlo | Análisis de sensibilidad y cuantificación de incertidumbre | Cap. 5, 6 |
| Precedentes (Salmanidou) | Validación metodológica del pipeline SPH → GP → UQ | Cap. 2, 6 |

---

## 2.10 Referencias del Capítulo

- Crespo, A.J.C., Gómez-Gesteira, M., & Dalrymple, R.A. (2007). Boundary conditions generated by dynamic particles in SPH methods. *CMC-Computers, Materials & Continua*, 5(3), 173-184.
- Crespo, A.J.C., et al. (2015). DualSPHysics: Open-source parallel CFD solver based on Smoothed Particle Hydrodynamics (SPH). *Computer Physics Communications*, 187, 204-216.
- Dehnen, W. & Aly, H. (2012). Improving convergence in smoothed particle hydrodynamics simulations without pairing instability. *Monthly Notices of the Royal Astronomical Society*, 425(2), 1068-1082.
- Domínguez, J.M., et al. (2022). DualSPHysics: from fluid dynamics to multiphysics problems. *Computational Particle Mechanics*, 9, 867-895.
- Engel, M. & May, S.M. (2012). Bonaire's boulder fields revisited: Evidence for Holocene tsunami impact on the Leeward Antilles. *Quaternary Science Reviews*, 54, 126-141.
- Etienne, S., et al. (2011). The use of boulders for characterising past tsunamis: Lessons from the 2004 Indian Ocean and 2009 South Pacific tsunamis. *Earth-Science Reviews*, 107(1-2), 76-90.
- Fourtakas, G., et al. (2019). Local uniform stencil (LUST) boundary condition for arbitrary 3-D boundaries in parallel smoothed particle hydrodynamics (SPH) models. *Computers & Fluids*, 190, 346-361.
- Gingold, R.A. & Monaghan, J.J. (1977). Smoothed particle hydrodynamics: Theory and application to non-spherical stars. *Monthly Notices of the Royal Astronomical Society*, 181(3), 375-389.
- Goto, K., et al. (2010). New insights of tsunami hazard from the 2011 Tohoku-oki event. *Marine Geology*, 290(1-4), 46-50.
- Imamura, F., et al. (2008). Tsunami modelling — manual and guides. IOC Manuals and Guides No. 56.
- Lucy, L.B. (1977). A numerical approach to the testing of the fission hypothesis. *The Astronomical Journal*, 82, 1013-1024.
- Molteni, D. & Colagrossi, A. (2009). A simple procedure to improve the pressure evaluation in hydrodynamic context using the SPH. *Computer Physics Communications*, 180(6), 861-872.
- Monaghan, J.J. (1992). Smoothed particle hydrodynamics. *Annual Review of Astronomy and Astrophysics*, 30(1), 543-574.
- Monaghan, J.J. (1994). Simulating free surface flows with SPH. *Journal of Computational Physics*, 110(2), 399-406.
- Monaghan, J.J. & Lattanzio, J.C. (1985). A refined particle method for astrophysical problems. *Astronomy and Astrophysics*, 149, 135-143.
- Nandasena, N.A.K., Paris, R., & Tanaka, N. (2011). Reassessment of hydrodynamic equations: Minimum flow velocity to initiate boulder transport by high energy events (storms, tsunamis). *Marine Geology*, 281(1-4), 70-84.
- Nandasena, N.A.K. & Tanaka, N. (2013). Boulder transport by high energy: numerical model-fitting experimental observations. *Ocean Engineering*, 57, 163-179.
- Nott, J. (2003). Waves, coastal boulder deposits and the importance of the pre-transport setting. *Earth and Planetary Science Letters*, 210(1-2), 269-276.
- Oakley, J.E. & O'Hagan, A. (2004). Probabilistic sensitivity analysis of complex models: a Bayesian approach. *Journal of the Royal Statistical Society: Series B*, 66(3), 751-769.
- Oetjen, J., et al. (2020). Significance of boulder transport for mitigating coastal hazard and risk. *Coastal Engineering*, 162, 103734.
- Rasmussen, C.E. & Williams, C.K.I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- Saltelli, A. (2002). Making best use of model evaluations to compute sensitivity indices. *Computer Physics Communications*, 145(2), 280-297.
- Salmanidou, D.M., et al. (2017). Statistical emulation of landslide-induced tsunamis. *Proceedings of the Royal Society A*, 473(2200), 20170026.
- Salmanidou, D.M., et al. (2020). Probabilistic Tsunami Hazard Analysis: High Performance Computing for Massive Scale Inundation Simulations. *Water*, 12(2), 416.
- Sarri, A., et al. (2012). Statistical emulation of a tsunami model for sensitivity analysis and uncertainty quantification. *Natural Hazards and Earth System Sciences*, 12(6), 2003-2018.
- Sobol', I.M. (2001). Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates. *Mathematics and Computers in Simulation*, 55(1-3), 271-280.
- Tasora, A., et al. (2016). Chrono: An open source multi-physics dynamics engine. In *High Performance Computing in Science and Engineering*, pp. 19-49. Springer.
- Violeau, D. & Rogers, B.D. (2016). Smoothed particle hydrodynamics (SPH) for free-surface flows: past, present and future. *Journal of Hydraulic Research*, 54(1), 1-26.
- Wendland, H. (1995). Piecewise polynomial, positive definite and compactly supported radial functions of minimal degree. *Advances in Computational Mathematics*, 4(1), 389-396.
