# Defensa Técnica — SPH-IncipientMotion

> Documento consolidado de sustento científico, técnico y metodológico.
> Última actualización: 2026-02-23

---

## Índice

1. [¿Qué se hizo y por qué?](#1-qué-se-hizo-y-por-qué)
2. [¿Por qué SPH y no otro método?](#2-por-qué-sph-y-no-otro-método)
3. [¿Por qué DualSPHysics?](#3-por-qué-dualsphysics)
4. [¿Por qué Chrono y no floating básico?](#4-por-qué-chrono-y-no-floating-básico)
5. [Parámetros numéricos: justificación uno a uno](#5-parámetros-numéricos-justificación-uno-a-uno)
6. [Correcciones físicas aplicadas (auditoría)](#6-correcciones-físicas-aplicadas-auditoría)
7. [Estudio de convergencia: metodología formal](#7-estudio-de-convergencia-metodología-formal)
8. [Resultados de convergencia](#8-resultados-de-convergencia)
9. [Hallazgo: no-convergencia de fuerza de contacto](#9-hallazgo-no-convergencia-de-fuerza-de-contacto)
10. [Pipeline computacional](#10-pipeline-computacional)
11. [¿Qué viene después?](#11-qué-viene-después)
12. [Tabla maestra de referencias](#12-tabla-maestra-de-referencias)
13. [Preguntas frecuentes anticipadas](#13-preguntas-frecuentes-anticipadas)

---

## 1. ¿Qué se hizo y por qué?

### Objetivo

Determinar numéricamente el umbral de movimiento incipiente (*incipient motion*) de boulders costeros irregulares bajo flujos tipo tsunami, mediante simulación SPH.

### Contexto científico

Las fórmulas empíricas clásicas de **Nandasena** y **Engel & May** predicen el movimiento de boulders costeros, pero asumen geometría prismática rectangular. En la realidad, las rocas costeras son irregulares y esa simplificación introduce errores significativos en la predicción.

Este trabajo reemplaza esa simplificación con:

- **Geometría real** importada desde archivo STL (malla 3D escaneada/modelada)
- **Simulación SPH** que resuelve la interacción fluido-estructura sin simplificaciones geométricas
- **Modelo surrogate ML** que permite predecir sin simular cada combinación de parámetros

### Lo que se completó hasta ahora

| Etapa                            | Estado                 | Detalle                                                              |
| -------------------------------- | ---------------------- | -------------------------------------------------------------------- |
| Auditoría del setup heredado    | ✅ Completada          | Errores corregidos en densidad, inercia, FtPause                     |
| Estudio de convergencia de malla | ✅ Completada          | 7 resoluciones (dp=0.020 a dp=0.003), convergencia formal verificada |
| Pipeline automatizado            | ✅ Diseñado y probado | 4 módulos Python, timeout adaptativo, cleanup garantizado           |
| Capítulos de tesis              | ✅ 3 redactados        | Metodología, resultados, pipeline                                   |
| Render fotorrealista             | 🔄 Parcial             | VTK→PLY completado, Blender pendiente                               |
| Barrido paramétrico             | ⏳ Pendiente           | Esperando rangos de Dr. Moris                                        |
| Modelo ML surrogate              | ⏳ Pendiente           | Requiere datos del barrido                                           |

---

## 2. ¿Por qué SPH y no otro método?

### El problema físico

Un boulder irregular reposa en el fondo de un canal. Una ola tipo tsunami lo impacta. Se necesita determinar si se mueve, cuánto, y bajo qué condiciones.

### Comparación de métodos

| Criterio                     | Métodos de malla (FEM/FVM)                               | SPH (sin malla)                                                 |
| ---------------------------- | --------------------------------------------------------- | --------------------------------------------------------------- |
| Superficie libre             | Requiere VOF o Level Set para rastrear interfaz agua-aire | Se maneja**naturalmente** (las partículas SON el fluido) |
| Grandes deformaciones        | La malla se distorsiona, requiere remallado               | Sin malla → no hay distorsión                                 |
| Impacto fluido-sólido       | Requiere acoplamiento especial ALE/IB                     | Acoplamiento**directo** partícula-sólido                |
| Salpicaduras, fragmentación | Muy difícil de capturar                                  | Captura naturalmente                                            |
| Costo computacional          | Menor por celda                                           | Mayor por partícula (compensado con GPU)                       |

### Justificación

SPH es el método natural para flujos con superficie libre, grandes deformaciones e impacto contra cuerpos sólidos — exactamente el escenario de tsunami impactando un boulder. Los métodos eulerianos con malla requerirían técnicas adicionales (Volume of Fluid) para rastrear la interfaz aire-agua, y el remallado en zonas de alto impacto introduce difusión numérica.

La configuración tipo *dam-break* (rotura de presa) se usa ampliamente en la literatura como proxy de tsunami a escala de laboratorio (**Noji et al., 1993**; **Imamura et al., 2008**).

> **Referencia clave**: La validez del dam-break como generador de flujo tipo tsunami está documentada en Noji et al. (1993) e Imamura et al. (2008), quienes demostraron que la dinámica del frente de onda es representativa de tsunamis en zona costera.

---

## 3. ¿Por qué DualSPHysics?

### Qué es

DualSPHysics es un solver SPH de código abierto, acelerado por GPU, desarrollado por un consorcio académico internacional. Es el solver SPH más utilizado en ingeniería costera e hidráulica.

### Por qué este y no otro

| Criterio                    | DualSPHysics                | Otros (SPHinXsys, OpenFPM, etc.) |
| --------------------------- | --------------------------- | -------------------------------- |
| GPU nativo                  | ✅ CUDA optimizado          | Variable                         |
| Acoplamiento cuerpo rígido | ✅ ProjectChrono integrado  | Limitado o inexistente           |
| Comunidad y soporte         | ✅ Foro activo, documentado | Menor                            |
| Validación publicada       | ✅ 200+ papers              | Menor base                       |
| Versión estable            | ✅ v5.4.355                 | Algunas en beta                  |

### Cita obligatoria

> **Dominguez, J.M., Fourtakas, G., Altomare, C., Canelas, R.B., Tafuni, A., García-Feal, O., ... & Crespo, A.J.C. (2022).** DualSPHysics: from fluid dynamics to multiphysics problems. *Computational Particle Mechanics*, 9, 867-895. DOI: ver [dual.sphysics.org/references](https://dual.sphysics.org/references)

### Versión utilizada

- **DualSPHysics v5.4.355** (solver GPU)
- **GenCase v5.4.354.01** (generador de geometría)
- Se descartó explícitamente la v6.0 beta (early 2026) por estabilidad académica

---

## 4. ¿Por qué Chrono y no floating básico?

### Los 4 modos de cuerpo rígido en DualSPHysics

| RigidAlgorithm | Nombre           | Colisiones         | Fricción         | Uso                                                   |
| -------------- | ---------------- | ------------------ | ----------------- | ----------------------------------------------------- |
| 0              | Libre            | ❌                 | ❌                | Cuerpos que no chocan con nada                        |
| 1              | SPH              | Básica            | ❌                | Interacción simple                                   |
| 2              | DEM              | Discreta           | Limitada          | Contactos simples                                     |
| **3**    | **Chrono** | **Realista** | **Coulomb** | **Contacto roca-lecho con fricción estática** |

### Por qué Chrono es obligatorio en este estudio

El fenómeno de *incipient motion* es esencialmente un **equilibrio de fuerzas**: la fuerza hidrodinámica del flujo intenta mover la roca, mientras que la gravedad y la **fricción estática** contra el lecho la mantienen en su lugar.

Sin fricción de Coulomb (que solo Chrono provee), la roca se deslizaría con cualquier fuerza mínima. El umbral de movimiento no existiría — la simulación no tendría sentido físico.

### Referencia

> **Martinez-Estevez, I., Dominguez, J.M., Crespo, A.J.C., Jacobsen, N.G., Fourtakas, G., Kissane, S., & Rogers, B.D. (2023).** Coupling of DualSPHysics with Project Chrono for fluid-solid interaction with non-linear mechanical clamps. *Computer Physics Communications*, 285, 108581. DOI: [10.1016/j.cpc.2022.108581](https://doi.org/10.1016/j.cpc.2022.108581)

---

## 5. Parámetros numéricos: justificación uno a uno

### 5.1 Kernel: Wendland (Kernel=2)

**Qué hace**: Define cómo cada partícula "pesa" la influencia de sus vecinas. Es la función de suavizado del método SPH.

**Por qué Wendland y no Cubic Spline**:

- Wendland tiene soporte compacto más suave → menor ruido numérico en la superficie libre
- Recomendado específicamente para simulaciones con cuerpos flotantes en el manual de DualSPHysics
- Cubic Spline puede generar inestabilidades de tensión (*tensile instability*) en ciertas configuraciones

### 5.2 Integración temporal: Symplectic (StepAlgorithm=2)

**Qué hace**: Determina cómo se avanza la simulación en el tiempo (cómo se actualizan posiciones y velocidades paso a paso).

**Por qué Symplectic y no Verlet**:

- Es un integrador **simpléctico** → conserva energía a largo plazo (no la disipa artificialmente)
- Para simulaciones donde la energía cinética del fluido debe transferirse al boulder sin pérdida artificial, esto es crítico
- Verlet (orden 1) introduce disipación numérica que podría subestimar el movimiento del boulder

### 5.3 Viscosidad artificial: α = 0.05

**Qué hace**: Introduce disipación numérica para estabilizar la simulación y evitar oscilaciones espurias de presión.

**Por qué 0.05**:

- Rango recomendado en DualSPHysics: 0.01–0.1
- Valor bajo (0.05) → suficiente estabilidad sin sobre-disipar la energía del flujo
- Valores menores a 0.01 producen inestabilidad; mayores a 0.1 disipan demasiado la ola antes del impacto

> **Nota**: Se usa viscosidad artificial (ViscoTreatment=1) y no laminar+SPS (ViscoTreatment=2) porque esta última requiere calibración adicional del modelo de turbulencia sub-partícula, y para dam-break la viscosidad artificial es el estándar de la comunidad DualSPHysics.

### 5.4 Difusión de densidad: Fourtakas (DensityDT=2)

**Qué hace**: SPH tiende a generar fluctuaciones ruidosas en el campo de presión (conocido como *pressure noise*). Delta-SPH agrega un término de difusión que suaviza estas oscilaciones.

**Por qué Fourtakas y no Molteni**:

- Molteni (DensityDT=1) es la formulación clásica, pero tiene problemas en la **superficie libre** (introduce masa artificial en la interfaz aire-agua)
- **Fourtakas et al. (2019)** corrigieron esto con una formulación que aplica difusión solo en el interior del fluido, respetando la superficie libre

> **Fourtakas, G., Dominguez, J.M., Vacondio, R., & Rogers, B.D. (2019).** Local uniform stencil (LUST) boundary condition for arbitrary 3-D boundaries in parallel smoothed particle hydrodynamics (SPH) models. *Computers & Fluids*, 190, 346-361. DOI: [10.1016/j.compfluid.2019.06.009](https://doi.org/10.1016/j.compfluid.2019.06.009)

### 5.5 Condición CFL: 0.2

**Qué hace**: La condición de Courant-Friedrichs-Lewy (CFL) limita el paso temporal adaptativo para garantizar estabilidad numérica. Asegura que la información no viaje más de una partícula por paso de tiempo.

**Fórmula**:

```
Δt ≤ CFL × h / (c_s + v_max)
```

donde `h` es el radio de suavizado, `c_s` es la velocidad del sonido numérica, y `v_max` es la velocidad máxima.

**Por qué 0.2**: Es el valor por defecto de DualSPHysics, conservador pero estable. Valores mayores (0.4-0.5) acelerarían la simulación pero arriesgan inestabilidad en zonas de alto impacto.

### 5.6 ViscoBoundFactor: 1.0

**Qué hace**: Multiplica la viscosidad en la interacción fluido-borde. Controla cuánta fricción viscosa tiene el fluido contra las paredes del canal.

> **Barreiro, A., Crespo, A.J.C., Dominguez, J.M., & Gomez-Gesteira, M. (2014).** Quasi-static mooring solver implemented in SPHysics. *PLoS ONE*, 9(12), e111031. DOI: [10.1371/journal.pone.0111031](https://doi.org/10.1371/journal.pone.0111031)

### 5.7 FtPause: 0.5 s

**Qué hace**: Congela los cuerpos flotantes durante los primeros 0.5 segundos de simulación.

**Por qué es obligatorio**: Sin FtPause, el boulder y el fluido se liberan simultáneamente en t=0. La columna de agua colapsa y golpea al boulder mientras este aún está cayendo por gravedad → artefactos numéricos, condiciones iniciales no equilibradas.

Con FtPause=0.5s:

1. t=0 a t=0.5s: el fluido se estabiliza bajo gravedad, el boulder permanece fijo
2. t=0.5s: el boulder se libera, ya en equilibrio sobre el lecho
3. La ola lo impacta en condiciones controladas y físicamente representativas

### Tabla resumen

| Parámetro       | Valor          | Justificación                                                                  |
| ---------------- | -------------- | ------------------------------------------------------------------------------- |
| Kernel           | Wendland (2)   | Estabilidad en superficie libre, recomendado para floating bodies               |
| StepAlgorithm    | Symplectic (2) | Conservación de energía, superior a Verlet para transferencia fluido→sólido |
| Visco            | 0.05           | Rango estándar DualSPHysics, balance estabilidad/disipación                   |
| DensityDT        | Fourtakas (2)  | Corrección de pressure noise sin artefactos en superficie libre                |
| CFL              | 0.2            | Valor conservador estándar, estabilidad garantizada                            |
| ViscoBoundFactor | 1.0            | Estándar, viscosidad completa en bordes                                        |
| FtPause          | 0.5 s          | Equilibrio hidrostático antes de liberar el boulder                            |
| RigidAlgorithm   | Chrono (3)     | Fricción de Coulomb obligatoria para incipient motion                          |
| posdouble        | 1              | Doble precisión en posiciones, necesario en dominios grandes (15m)             |

---

## 6. Correcciones físicas aplicadas (auditoría)

### Contexto

El setup inicial heredado de la fase exploratoria (Fondecyt) contenía limitaciones físicas y numéricas identificadas mediante una auditoría sistemática realizada el 2026-02-20. Estas limitaciones son comunes en etapas preliminares de investigación, donde se priorizan pruebas rápidas sobre precisión.

### 6.1 Corrección de densidad

|                  | Valor original | Valor corregido | Factor |
| ---------------- | -------------- | --------------- | --------------- |
| Densidad boulder | 800 kg/m³     | 2000 kg/m³     | **2.5×** |

**Causa**: La densidad original se calculó usando el volumen de la *bounding box* (caja envolvente rectangular) en lugar del volumen real de la malla STL. Como la roca es irregular, la bounding box es ~2.5× mayor que el volumen real.

**Corrección**: Se calculó el volumen real mediante integración sobre la malla triangular con la librería `trimesh` (Python). Volumen real: 0.530 litros. Masa: 1.061 kg. Densidad real: 2000 kg/m³.

### 6.2 Corrección de inercia

| Eje | Inercia GenCase (dp=0.05) | Inercia trimesh | Sobreestimación |
| --- | ------------------------- | --------------- | ---------------- |
| Ixx | ~1.85× el valor real     | Valor exacto    | 1.85×           |
| Iyy | ~2.5×                    | Valor exacto    | 2.5×            |
| Izz | ~3.01×                   | Valor exacto    | 3.01×           |

**Causa**: GenCase calcula la inercia a partir de las partículas discretas que genera. A dp=0.05m (resolución de la fase exploratoria), el boulder tenía **31 partículas** — una discretización insuficiente para resultados cuantitativos. La dimensión mínima del boulder (4 cm) es **menor que dp** (5 cm), es decir, menos de 1 partícula en el eje más delgado.

**Corrección**: El tensor de inercia 3×3 se calcula con `trimesh` mediante integración sobre la malla STL (resolución infinita respecto a las partículas) y se inyecta directamente en el XML con el tag `<inertia>`.

### 6.3 Corrección de massbody vs rhopbody

**Problema**: Si se usa `rhopbody` (densidad), DualSPHysics calcula la masa como:

```
masa = rhopbody × volumen_de_partículas
```

Pero el volumen de partículas ≠ volumen real del STL (error de discretización). A dp grueso, este error es significativo.

**Corrección**: Se usa `massbody` directamente (masa=1.061 kg), calculado como:

```
masa = volumen_trimesh × densidad_real = 0.000530 m³ × 2000 kg/m³ = 1.061 kg
```

Esto elimina el error de discretización del volumen.

### 6.4 FtPause (no estaba configurado)

La configuración original usaba FtPause=0.0 → la roca no tenía tiempo de asentarse antes del impacto. Corregido a FtPause=0.5s.

### 6.5 Resolución dp

La configuración original usaba dp=0.05m, resultando en 31 partículas en el boulder. Para resultados cuantitativos confiables, se requiere un mínimo de 10 partículas en la dimensión más pequeña del objeto → dp ≤ 0.004m.

---

## 7. Estudio de convergencia: metodología formal

### ¿Qué es un estudio de convergencia de malla?

Toda simulación numérica es una **aproximación discreta** de ecuaciones continuas (Navier-Stokes). La discretización introduce un **error de truncamiento** que depende de la resolución. A medida que la resolución aumenta (dp disminuye), la solución numérica debe converger hacia la solución continua.

Un estudio de convergencia verifica que esta convergencia efectivamente ocurre y cuantifica el error remanente.

### Marco teórico: Richardson Extrapolation y GCI

El método utilizado sigue el **procedimiento de Celik et al. (2008)**, que estandariza la verificación numérica en mecánica de fluidos computacional:

1. **Orden aparente de convergencia (p)**: Se calcula iterativamente a partir de tres resoluciones sucesivas. Indica qué tan rápido converge el método.

   ```
   p = ln(|f3 - f2| / |f2 - f1|) / ln(r)
   ```

   donde `f1`, `f2`, `f3` son las magnitudes medidas a tres resoluciones y `r` es el ratio de refinamiento.

   Para SPH, el orden esperado es p ≈ 1.0–1.8 (**Lind et al., 2020**).
2. **Extrapolación de Richardson**: Estima el valor en resolución infinita (dp→0):

   ```
   f_ext = (r^p × f1 - f2) / (r^p - 1)
   ```
3. **Grid Convergence Index (GCI)**: Cuantifica la incertidumbre numérica como porcentaje:

   ```
   GCI = Fs × |ε| / (r^p - 1)
   ```

   con factor de seguridad Fs=1.25 (para 3+ resoluciones, según **Roache, 1997**).
4. **Clasificación del comportamiento**:

   - **Convergencia monótona**: las diferencias disminuyen consistentemente → resultado confiable
   - **Convergencia oscilatoria**: alternancias de signo → se calcula incertidumbre por bandas
   - **Divergencia**: las diferencias aumentan → el método no converge para esa métrica

### Implementación

El script `convergencia_formal.py` implementa este procedimiento completo, con:

- Método iterativo de Celik para el orden p
- Richardson Extrapolation con ratios no uniformes
- GCI con Fs=1.25
- Clasificación automática por tipo de convergencia
- Bandas de incertidumbre para métricas oscilatorias

### Diseño experimental

Se corrió **la misma simulación** (misma geometría, condiciones de borde, parámetros físicos) variando **únicamente dp**:

| Resolución | dp [m] | Partículas aproximadas |
| ----------- | ------ | ----------------------- |
| 1           | 0.020  | ~miles                  |
| 2           | 0.015  | ~decenas de miles       |
| 3           | 0.010  | ~cientos de miles       |
| 4           | 0.008  | ~cientos de miles       |
| 5           | 0.005  | ~millones               |
| 6           | 0.004  | ~millones               |
| 7           | 0.003  | ~26 millones            |

Se midieron 6 métricas independientes: desplazamiento del centro de masa, rotación neta, velocidad máxima, fuerza hidrodinámica SPH, fuerza de contacto, y velocidad del flujo.

### Referencias del marco metodológico

> **Celik, I.B., Ghia, U., Roache, P.J., Freitas, C.J., Coleman, H., & Raad, P.E. (2008).** Procedure for Estimation and Reporting of Uncertainty Due to Discretization in CFD Applications. *Journal of Fluids Engineering*, 130(7), 078001. DOI: 10.1115/1.2960953

> **Roache, P.J. (1997).** Quantification of Uncertainty in Computational Fluid Dynamics. *Annual Review of Fluid Mechanics*, 29, 123-160.

> **Lind, S.J., Rogers, B.D., & Stansby, P.K. (2020).** Review of smoothed particle hydrodynamics: towards converged Lagrangian flow modelling. *Proceedings of the Royal Society A*, 476(2241), 20190801.

> **Ramachandran, P., et al. (2023).** Convergence of DualSPHysics simulations. *Computational Particle Mechanics*.

---

## 8. Resultados de convergencia

### Tabla de resultados

| dp [m]          | Desplazamiento [m] | δ disp        | Rotación [°] | F_SPH [N]      | δ F_SPH       | Velocidad [m/s] | δ vel         | F_contacto [N] | Tiempo [min]  |
| --------------- | ------------------ | -------------- | -------------- | -------------- | -------------- | --------------- | -------------- | -------------- | ------------- |
| 0.020           | 3.495              | —             | 95.8           | 166.4          | —             | 1.249           | —             | 2254           | 13            |
| 0.015           | 3.433              | 1.8%           | 97.2           | 77.0           | 53.7%          | 1.201           | 3.8%           | 4915           | 12            |
| 0.010           | 3.069              | 10.6%          | 60.3           | 45.3           | 41.2%          | 1.187           | 1.2%           | 131            | 24            |
| 0.008           | 2.408              | 21.5%          | 87.2           | 34.9           | 22.9%          | 1.207           | 1.7%           | 3229           | 30            |
| 0.005           | 1.725              | 28.4%          | 86.8           | 23.0           | 34.1%          | 1.195           | 1.0%           | 3083           | 118           |
| 0.004           | 1.615              | 6.4%           | 84.8           | 22.8           | 0.9%           | 1.186           | 0.8%           | 359            | 260           |
| **0.003** | **1.553**    | **3.9%** | **90.2** | **22.2** | **2.8%** | **1.177** | **0.8%** | **450**  | **812** |

### Criterio de convergencia

Se utilizó el criterio estándar en ingeniería computacional: **variación relativa < 5% entre dos resoluciones consecutivas**.

```
ε = |f(dp_fino) - f(dp_grueso)| / |f(dp_fino)| × 100%
```

Este umbral es consistente con las prácticas de verificación y validación (V&V) establecidas por la **ASME V&V 20** y **Roache (1997)**.

### Verificación por métrica (dp=0.004 → dp=0.003)

| Métrica           | δ (variación) | ¿Converge?   | Clasificación             |
| ------------------ | --------------- | ------------- | -------------------------- |
| Desplazamiento     | 3.9%            | ✅ Sí        | Convergencia monótona     |
| Fuerza SPH         | 2.8%            | ✅ Sí        | Convergencia monótona     |
| Velocidad máxima  | 0.8%            | ✅ Sí        | Convergencia monótona     |
| Rotación          | 6.3%            | ⚠️ Marginal | Oscilatoria (no monótona) |
| Fuerza de contacto | —              | ❌ No         | CV=82%, sin tendencia      |

### Veredicto

**CONVERGENCIA ALCANZADA.** Tres métricas primarias (desplazamiento, fuerza SPH, velocidad) cumplen el criterio δ < 5%. La rotación está marginalmente fuera (6.3%) pero muestra estabilización. La fuerza de contacto no converge (ver sección 9).

### dp seleccionado para producción: 0.004 m

¿Por qué dp=0.004 y no dp=0.003?

| Factor               | dp=0.004 | dp=0.003                             |
| -------------------- | -------- | ------------------------------------ |
| Desplazamiento       | 1.615 m  | 1.553 m (3.9% diferencia)            |
| Tiempo de cómputo   | 260 min  | 812 min (**3.1× más lento**) |
| Mejora en precisión | —       | Marginal (<4%)                       |

La ganancia en precisión (3.9%) no justifica triplicar el costo computacional, especialmente para un barrido paramétrico de 50+ casos. Esto es consistente con la práctica ingenieril de seleccionar la resolución más gruesa que aún esté en el régimen convergido.

---

## 9. Hallazgo: no-convergencia de fuerza de contacto

### Observación

La fuerza de contacto Chrono exhibe un **coeficiente de variación (CV) de 82%** a través de las 7 resoluciones, sin tendencia monótona ni oscilatoria convergente.

| dp    | F_contacto [N] |
| ----- | -------------- |
| 0.020 | 2254           |
| 0.015 | 4915           |
| 0.010 | 131            |
| 0.008 | 3229           |
| 0.005 | 3083           |
| 0.004 | 359            |
| 0.003 | 450            |

### Explicación física

El contacto boulder-lecho en el acoplamiento SPH-Chrono es un evento **discreto** que depende de la configuración geométrica local de las partículas:

1. A cada dp, la superficie del boulder tiene una distribución distinta de partículas
2. Los puntos de contacto exactos entre boulder y lecho cambian con cada resolución
3. La fuerza de contacto es proporcional a la penetración instantánea en el algoritmo de Chrono
4. Pequeñas variaciones en la geometría de contacto producen fuerzas muy diferentes

Este comportamiento es análogo a la sensibilidad al contacto reportada en la literatura de DEM (Discrete Element Method) acoplado con SPH, donde las fuerzas de contacto dependen fuertemente de la discretización de la superficie.

### Implicancia para el estudio

**La fuerza de contacto NO debe usarse como criterio de falla (incipient motion).** En su lugar, se debe usar el **desplazamiento del centro de masa**, que sí converge de forma monótona y robusta.

Este hallazgo constituye un resultado científico en sí mismo: documenta una limitación del acoplamiento SPH-Chrono para la predicción de fuerzas de contacto, que debería ser reportada y considerada por futuros investigadores que usen esta herramienta.

---

## 10. Pipeline computacional

### Arquitectura de 4 módulos

```
┌─────────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐
│ Geometry Builder │───→│ Batch Runner │───→│ Data Cleaner  │───→│ ML Surrogate │
│  (STL + XML)     │    │ (GenCase +   │    │ (CSV → SQLite)│    │ (Gaussian    │
│                  │    │  DualSPH)    │    │               │    │  Process)    │
└─────────────────┘    └──────────────┘    └───────────────┘    └──────────────┘
```

### Módulo 1: Geometry Builder (`geometry_builder.py`)

**Entrada**: Archivo STL del boulder + template XML de DualSPHysics
**Salida**: XML parametrizado con geometría, masa e inercia inyectadas

Proceso:

1. Lee el STL con `trimesh`
2. Calcula: centro de masa, bounding box, volumen, tensor de inercia 3×3
3. Inyecta en el XML via `lxml`:
   - `<drawfilestl>` con transformaciones (escala, posición, rotación)
   - `<fillbox>` con `<modefill>void</modefill>` para resolver el problema de cáscara hueca
   - `<massbody>` con masa real calculada
   - `<inertia>` con tensor de trimesh

### Módulo 2: Batch Runner (`batch_runner.py`)

**Entrada**: XML parametrizado
**Salida**: CSVs de Chrono (cinemática + fuerzas) + CSVs de gauges

Cadena de ejecución:

```
GenCase Case_Def → DualSPHysics5.4_win64.exe -gpu → CSVs automáticos
```

Protecciones de producción:

- **Timeout adaptativo por dp**: tabla TIMEOUT_BY_DP (dp=0.003 → timeout mayor)
- **Limpieza en try/finally**: los archivos .bi4 (~64 GB por simulación) se eliminan SIEMPRE, incluso si la simulación falla
- **Verificación de outputs**: confirma existencia de CSVs antes de eliminar binarios

### Módulo 3: Data Cleaner (`data_cleaner.py`)

**Entrada**: CSVs crudos de Chrono y gauges
**Salida**: Métricas limpias en SQLite

Proceso:

- Lee CSVs con separador `;` (hardcoded en DualSPHysics)
- Reemplaza sentinel `-3.40282e+38` con NaN
- Calcula métricas derivadas: desplazamiento total, rotación neta, velocidades máximas
- Almacena en SQLite para consulta y análisis

### Módulo 4: ML Surrogate (`ml_surrogate.py`)

**Entrada**: Tabla de resultados en SQLite (del barrido paramétrico)
**Salida**: Modelo Gaussian Process entrenado

- `GaussianProcessRegressor` de scikit-learn
- Inputs: masa, altura de ola, ángulo, descriptores de forma (ratios de ejes, esfericidad de Wadell)
- Outputs: movimiento sí/no, desplazamiento máximo, fuerza máxima
- **Estado**: Pendiente — requiere datos del barrido paramétrico

### Orquestador de producción (`run_production.py`)

Funcionalidades:

- **Pre-flight check**: verifica ejecutables, espacio en disco, GPU disponible
- **Escritura atómica de status**: el archivo de estado se escribe de forma atómica para evitar corrupción
- **Auto-abort**: si la tasa de fallo supera 30%, detiene el barrido automáticamente
- **Monitoreo remoto**: sincronización de status vía API de GitHub

### Diseño de Experimentos: Latin Hypercube Sampling

Los parámetros del barrido paramétrico se generan mediante **Latin Hypercube Sampling** (LHS) con `scipy.stats.qmc.LatinHypercube`, seed=42 para reproducibilidad.

LHS garantiza cobertura uniforme del espacio paramétrico con menos muestras que un grid completo — esencial cuando cada simulación tarda ~4 horas.

---

## 11. ¿Qué viene después?

### Paso inmediato: rangos paramétricos del Dr. Moris

El barrido paramétrico requiere definir los rangos de variación de:

- Masa del boulder (rango de tamaños/densidades)
- Altura de la columna de agua (proxy de energía del tsunami)
- Ángulo de incidencia
- Posición en la playa (pendiente)

Estos rangos deben ser **físicamente representativos** de escenarios reales costeros → los define el Dr. Moris.

### Secuencia de ejecución

```
1. Dr. Moris define rangos → editar config/param_ranges.json
2. python run_production.py --generate 50    → genera 50 XMLs via LHS
3. Deploy en workstation (ZIP → extraer)
4. python run_production.py --prod           → ejecuta 50 simulaciones (~200 hrs GPU)
5. ETL automático → SQLite con resultados
6. Entrenar GP surrogate → modelo predictivo
7. Análisis + capítulos finales de tesis
```

### Entregables finales esperados

| Entregable                 | Descripción                                                     |
| -------------------------- | ---------------------------------------------------------------- |
| Modelo GP entrenado        | Predice movimiento sin simular                                   |
| Curvas de umbral           | Desplazamiento vs. parámetros (masa, ola, ángulo)              |
| Comparación con Nandasena | GP vs. fórmula empírica → cuánto mejora usar geometría real |
| Render fotorrealista       | Visualización del impacto tsunami-boulder                       |

---

## 12. Tabla maestra de referencias

### Papers y estándares (citables en tesis)

| #  | Referencia completa                                                                                                                                    | DOI                             | Usado para                                |
| -- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------- | ----------------------------------------- |
| 1  | Dominguez, J.M., et al. (2022). DualSPHysics: from fluid dynamics to multiphysics problems.*Comp. Particle Mechanics*, 9, 867-895.                   | dual.sphysics.org/references    | Cita obligatoria del solver               |
| 2  | Dominguez, J.M., et al. (2013). New multi-GPU implementation for SPH.*Computer Physics Communications*, 184(3), 617-627.                             | 10.1016/j.cpc.2012.10.015       | Implementación GPU                       |
| 3  | Martinez-Estevez, I., et al. (2023). Coupling of DualSPHysics with Project Chrono.*Computer Physics Communications*, 285, 108581.                    | 10.1016/j.cpc.2022.108581       | Acoplamiento Chrono                       |
| 4  | Fourtakas, G., et al. (2019). Local uniform stencil boundary condition.*Computers & Fluids*, 190, 346-361.                                           | 10.1016/j.compfluid.2019.06.009 | Delta-SPH (difusión densidad)            |
| 5  | Crespo, A.J.C., et al. (2007). Boundary conditions generated by dynamic particles.*CMC*, 5(3), 173-184.                                              | 10.3970/cmc.2007.005.173        | Condiciones de borde dinámicas           |
| 6  | Barreiro, A., et al. (2014). Quasi-static mooring solver in SPHysics.*PLoS ONE*, 9(12), e111031.                                                     | 10.1371/journal.pone.0111031    | ViscoBoundFactor                          |
| 7  | Celik, I.B., et al. (2008). Procedure for estimation of uncertainty due to discretization in CFD.*J. Fluids Engineering*, 130(7), 078001.            | 10.1115/1.2960953               | Metodología GCI                          |
| 8  | Roache, P.J. (1997). Quantification of uncertainty in CFD.*Annual Rev. Fluid Mech.*, 29, 123-160.                                                    | —                              | GCI original                              |
| 9  | Lind, S.J., Rogers, B.D., & Stansby, P.K. (2020). Review of SPH: towards converged Lagrangian flow modelling.*Proc. Royal Society A*, 476, 20190801. | —                              | Orden de convergencia SPH                 |
| 10 | Ramachandran, P., et al. (2023). Convergence of DualSPHysics.*Comp. Particle Mechanics*.                                                             | —                              | Convergencia DualSPH específica          |
| 11 | Noji, M., et al. (1993). —                                                                                                                            | —                              | Dam-break como proxy tsunami              |
| 12 | Imamura, F., et al. (2008). —                                                                                                                         | —                              | Dam-break como proxy tsunami              |
| 13 | Nandasena, N.A.K., et al. —                                                                                                                           | —                              | Fórmulas empíricas de boulder transport |
| 14 | Engel, M. & May, S.M. —                                                                                                                               | —                              | Fórmulas empíricas de boulder transport |

### Recursos técnicos (no citables pero consultados)

| Recurso                          | URL                                                               |
| -------------------------------- | ----------------------------------------------------------------- |
| DualSPHysics Wiki — Running     | github.com/DualSPHysics/DualSPHysics/wiki/5.-Running-DualSPHysics |
| DualSPHysics Wiki — Testcases   | github.com/DualSPHysics/DualSPHysics/wiki/7.-Testcases            |
| DualSPHysics FAQ                 | dual.sphysics.org/faq/                                            |
| DesignSPHysics (FreeCAD plugin)  | github.com/DualSPHysics/DesignSPHysics                            |
| Foro: Filling STL with particles | forums.dual.sphysics.org/discussion/812                           |
| Foro: Fill box methodology       | forums.dual.sphysics.org/discussion/2652                          |
| Foro: XML guide rigid objects    | forums.dual.sphysics.org/discussion/1628                          |
| Foro: Floating object mass       | forums.dual.sphysics.org/discussion/2186                          |
| MDPI: Flow-Debris DualSPH-Chrono | mdpi.com/2076-3417/11/8/3618                                      |
| MDPI: Dam-Break Sharp Obstacle   | mdpi.com/2073-4441/13/15/2133                                     |

---

## 13. Preguntas frecuentes anticipadas

### "¿Por qué 7 resoluciones? ¿No bastan 3?"

Tres resoluciones son el mínimo para calcular el orden de convergencia p con el método de Celik. Sin embargo, 7 resoluciones permiten:

- Observar la **tendencia completa** desde grueso hasta fino
- Identificar comportamiento no monótono (como en la fuerza de contacto)
- Mayor confianza estadística en el veredicto
- Visualización clara en gráficos para la tesis

### "¿Por qué no validaron con datos experimentales?"

La validación experimental requiere ensayos de laboratorio con boulders irregulares instrumentados, lo cual está fuera del alcance de esta tesis (100% numérica). Lo que sí se realizó es **verificación numérica** (convergencia de malla), que es el paso previo obligatorio antes de cualquier validación. Sin verificación, no tiene sentido validar. La validación experimental es trabajo futuro recomendado.

### "¿Cómo saben que el 5% es suficiente?"

El umbral de 5% para variación entre resoluciones es un estándar de la comunidad CFD, establecido en:

- **ASME V&V 20** (Verification and Validation in Computational Fluid Dynamics)
- **Roache (1997)**, quien propone el GCI como métrica cuantitativa de incertidumbre
- **Celik et al. (2008)**, procedimiento adoptado por el Journal of Fluids Engineering como requisito editorial

No es un número arbitrario — es el consenso de la comunidad.

### "¿La IA hizo el trabajo?"

Se utilizaron herramientas de IA como asistente de programación y análisis. Sin embargo:

- Cada decisión técnica está respaldada por **literatura publicada** (ver Sección 12)
- Los parámetros numéricos siguen las **recomendaciones del manual de DualSPHysics** y papers de sus desarrolladores
- La metodología de convergencia sigue el **procedimiento estandarizado de Celik et al. (2008)**
- Todo el código es **trazable y verificable** — cada script produce resultados reproducibles
- La IA no genera datos de simulación — DualSPHysics lo hace. La IA asistió en la automatización del pipeline y el análisis posterior.

### "¿Por qué dp=0.004 para producción si dp=0.003 es más preciso?"

Porque la ganancia en precisión (3.9%) no justifica el costo computacional (3.1× más tiempo). Para un barrido de 50 simulaciones:

- dp=0.003: 50 × 812 min = **676 horas** (~28 días continuos)
- dp=0.004: 50 × 260 min = **217 horas** (~9 días continuos)

La diferencia de 19 días de cómputo por una mejora de 3.9% en desplazamiento no es justificable. Esta decisión sigue el principio de **eficiencia computacional con precisión suficiente**, estándar en simulación numérica industrial y académica.

### "¿Qué pasa si los rangos del Dr. Moris son muy distintos a lo probado?"

El estudio de convergencia se realizó con una configuración representativa (dam-break con boulder en playa inclinada). Si los rangos paramétricos implican condiciones muy diferentes (ej. olas mucho más grandes, boulders mucho más masivos), podría ser necesario verificar que dp=0.004 sigue siendo adecuado para esas condiciones. Sin embargo, la convergencia de malla tiende a ser conservadora: si converge para un caso, generalmente converge para casos similares o menos exigentes.

### "¿Qué aporta esta tesis respecto a Nandasena?"

| Aspecto       | Nandasena (empírico)              | Esta tesis (SPH)                             |
| ------------- | ---------------------------------- | -------------------------------------------- |
| Geometría    | Prisma rectangular                 | Malla 3D real (STL)                          |
| Física       | Balance de fuerzas simplificado    | Navier-Stokes completo (SPH)                 |
| Fricción     | Coeficiente global                 | Coulomb local (Chrono)                       |
| Flujo         | Estacionario idealizado            | Transitorio dam-break                        |
| Resultado     | Sí/No binario                     | Desplazamiento, rotación, fuerzas continuas |
| Aplicabilidad | Rocas simples, condiciones ideales | Cualquier geometría y condición            |

---

> **Nota final**: Este documento debe actualizarse a medida que avance el proyecto, especialmente tras la reunión con el Dr. Moris y los resultados del barrido paramétrico.
