# Capítulo 1: Introducción

## 1.1 Contexto General

Los tsunamis constituyen una de las amenazas naturales más destructivas que afectan a las zonas costeras. Además de los daños directos a infraestructura y vidas humanas, estos eventos son capaces de transportar bloques de roca de varias toneladas desde acantilados costeros hacia el interior, depositándolos a distancias considerables de la línea de costa. Estos depósitos de bloques costeros (*coastal boulder deposits*, CBD) representan registros geológicos invaluables de eventos extremos pasados, permitiendo estimar la magnitud de tsunamis prehistóricos cuando no existen registros instrumentales.

En la costa de Bahía Cisnes, Región de Atacama (Chile), se han identificado bloques de hasta 40 toneladas transportados por un evento de tsunami ocurrido hace aproximadamente 600 años (Moris et al., en preparación). La reconstrucción de las condiciones hidrodinámicas necesarias para movilizar estos bloques — es decir, la determinación de las velocidades y alturas de flujo críticas — es fundamental tanto para la evaluación del riesgo sísmico-tsunámico como para la comprensión de la dinámica costera de largo plazo.

## 1.2 Planteamiento del Problema

Las ecuaciones predictivas clásicas para el transporte de bloques costeros por tsunami (Nott, 2003; Nandasena et al., 2011; Engel & May, 2012) relacionan las fuerzas hidrodinámicas del flujo con las fuerzas resistentes del bloque (peso, fricción, inercia) para determinar si el movimiento ocurre. Sin embargo, estas formulaciones comparten una simplificación fundamental: **tratan los bloques como prismas rectangulares homogéneos**.

Esta idealización tiene consecuencias significativas:

1. **Subestimación de la variabilidad geométrica:** Los bloques costeros reales presentan geometrías irregulares con bordes asimétricos, concavidades y distribuciones de masa no uniformes. Un prisma rectangular no captura los brazos de palanca reales que determinan la estabilidad rotacional.

2. **Coeficientes de arrastre genéricos:** Las ecuaciones emplean coeficientes de arrastre (C_D) y sustentación (C_L) calibrados para geometrías simples, que no reflejan la interacción compleja entre un flujo turbulento y una superficie irregular.

3. **Modos de fallo reducidos:** Las formulaciones analíticas consideran modos de fallo aislados (deslizamiento, volcamiento, sustentación), mientras que en la realidad los bloques experimentan combinaciones simultáneas de traslación y rotación.

4. **Ausencia de la dinámica transitoria:** Las ecuaciones evalúan un equilibrio estático instantáneo, ignorando la evolución temporal del impacto — la aceleración del flujo, la respuesta dinámica del bloque, y las fuerzas de contacto con el sustrato.

La pregunta central que motiva esta investigación es: **¿Cómo influye la forma geométrica irregular de un bloque costero en las condiciones críticas de flujo necesarias para iniciar su movimiento?**

## 1.3 Justificación

La simulación numérica ofrece una alternativa para superar las limitaciones de las ecuaciones analíticas. El método de Hidrodinámica de Partículas Suavizadas (SPH) es particularmente adecuado para este problema porque:

- **No requiere malla:** SPH discretiza el fluido en partículas que se mueven libremente, lo que permite simular de forma natural la ruptura de olas, salpicaduras y el impacto violento del flujo contra el bloque — fenómenos que causan problemas graves en métodos basados en malla (Violeau & Rogers, 2016).

- **Maneja geometrías complejas:** Los bloques irregulares se representan directamente como nubes de partículas generadas a partir de modelos 3D (archivos STL), sin necesidad de generar mallas conformes alrededor de geometrías complicadas.

- **Acoplamiento fluido-estructura:** A través de la integración con motores de cuerpos rígidos como ProjectChrono, SPH resuelve simultáneamente la dinámica del fluido y el movimiento del bloque (traslación, rotación, colisiones con el sustrato).

- **Aceleración GPU:** Implementaciones modernas como DualSPHysics (Domínguez et al., 2022) aprovechan GPUs NVIDIA para simular decenas de millones de partículas en tiempos razonables, haciendo viable la realización de campañas paramétricas.

Sin embargo, cada simulación SPH de alta resolución requiere horas de cómputo GPU, lo que impide explorar exhaustivamente el espacio de parámetros (altura de flujo, masa del bloque, orientación, forma) mediante fuerza bruta. Para resolver esta limitación, se emplea un **modelo surrogate** basado en regresión de procesos gaussianos (GP), que — entrenado con un número limitado de simulaciones — permite predecir resultados en milisegundos y realizar análisis de sensibilidad global.

## 1.4 Hipótesis

La forma geométrica irregular de un bloque costero modifica significativamente las condiciones de flujo críticas para el inicio de su movimiento, en comparación con las predicciones obtenidas al aproximar el bloque como un prisma rectangular equivalente. Específicamente, se postula que:

1. El umbral de movimiento incipiente (altura de flujo crítica) depende de la orientación del bloque respecto al flujo.
2. La irregularidad geométrica introduce modos de fallo combinados (traslación + rotación) que las ecuaciones analíticas no capturan.
3. Un modelo surrogate entrenado con simulaciones SPH puede predecir las condiciones de movimiento incipiente con incertidumbre cuantificada.

## 1.5 Objetivos

### 1.5.1 Objetivo General

Determinar numéricamente las condiciones críticas de flujo para el movimiento incipiente de bloques costeros irregulares bajo cargas tipo tsunami, mediante simulaciones SPH acopladas con dinámica de cuerpos rígidos y modelación surrogate.

### 1.5.2 Objetivos Específicos

1. **Desarrollar un pipeline computacional automatizado** que integre la generación de geometría, la ejecución de simulaciones SPH-Chrono en GPU, la extracción de datos cinemáticos, y el entrenamiento de un modelo surrogate de procesos gaussianos.

2. **Verificar la resolución numérica** mediante un estudio de convergencia de malla que garantice resultados independientes de la discretización espacial.

3. **Realizar una campaña paramétrica** de screening (50 simulaciones) variando altura de columna de agua, masa del bloque y orientación, para identificar los parámetros dominantes mediante análisis de sensibilidad global (índices de Sobol).

4. **Entrenar y validar un modelo surrogate** (proceso gaussiano) que prediga el desplazamiento máximo del bloque a partir de los parámetros de entrada, con cuantificación de incertidumbre.

5. **Contribuir al conocimiento** sobre las limitaciones del acoplamiento SPH-Chrono para la predicción de fuerzas de contacto, documentando el hallazgo de no-convergencia observado en el estudio de malla.

## 1.6 Marco Institucional

Esta investigación se desarrolla como trabajo de titulación en la Facultad de Ciencias de la Ingeniería y Construcción (FaCIC) de la Universidad Católica del Norte (UCN), Antofagasta, Chile. Se enmarca dentro del proyecto Fondecyt de Iniciación 2025 del Dr. Joaquín Moris Barra, cuyo objetivo es mejorar las ecuaciones predictivas de transporte incipiente de bloques costeros incorporando la variable de forma geométrica. El proyecto Fondecyt contempla tanto experimentos de laboratorio como simulaciones numéricas; la presente tesis constituye la componente numérica.

## 1.7 Alcance y Limitaciones

**Alcance:**
- El estudio es 100% numérico — no incluye trabajo experimental de laboratorio.
- Se emplea una única geometría de bloque (modelo BLIR3) para la campaña paramétrica de screening.
- El flujo se genera mediante dam-break (rotura de presa), representativo de un bore de tsunami a escala de laboratorio.
- Los parámetros de fricción corresponden a una configuración de laboratorio (PVC sobre acero).

**Limitaciones:**
- Los rangos de parámetros (altura de columna, masa, rotación) son provisionales y están pendientes de validación con el director de tesis.
- La extrapolación a escala de campo requiere análisis dimensional que no se aborda en esta tesis.
- Se emplea una sola forma de bloque; la extensión a 7 formas está planificada como trabajo futuro dentro del proyecto Fondecyt.
- Las propiedades de fricción de laboratorio (PVC-acero) no representan directamente condiciones de campo (caliza sobre roca).

## 1.8 Estructura de la Tesis

La tesis se organiza en 7 capítulos:

- **Capítulo 1 — Introducción:** Contexto, problema, objetivos y alcance (presente capítulo).
- **Capítulo 2 — Marco Teórico:** Fundamentos de transporte de bloques costeros, método SPH, acoplamiento con cuerpos rígidos, y modelación surrogate.
- **Capítulo 3 — Metodología Numérica:** Configuración del dominio computacional, parámetros del solver, instrumentación virtual, y estudio de convergencia de malla.
- **Capítulo 4 — Resultados del Estudio de Convergencia:** Análisis detallado de la convergencia por métrica, hallazgo sobre fuerza de contacto, y selección de resolución.
- **Capítulo 5 — Pipeline Computacional:** Arquitectura del pipeline automatizado, módulos, mecanismos de robustez, y diseño de experimentos.
- **Capítulo 6 — Resultados de la Campaña Paramétrica:** Análisis de los 50 casos de screening, modelo GP, análisis de sensibilidad (Sobol), y cuantificación de incertidumbre.
- **Capítulo 7 — Conclusiones y Trabajo Futuro:** Síntesis de hallazgos, contribuciones, limitaciones, y líneas de investigación futura.
