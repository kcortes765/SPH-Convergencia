# PROJECT DNA & MASTER BRIEF
> Ultima actualizacion: 2026-02-24

## 1. IDENTIFICACION Y PERFIL DEL LIDER DEL PROYECTO

* **Investigador Principal:** Kevin Cortes Hernandez (Sebastian Cortes para fines de branding internacional).
* **Institucion:** Universidad Catolica del Norte (UCN), Antofagasta, Chile.
* **Situacion Academica:** Alumno de 5to ano de Ingenieria Civil, cursando Capstone Project (2026).
* **Mentor Academico:** Dr. Joaquin Moris Barra (Secretario de Investigacion FaCIC / Investigador Principal Fondecyt Iniciacion).
* **Perfil Tecnico:** Computational Civil Engineer & Data Specialist. Experto en automatizacion de procesos de ingenieria, surrogate modeling con Gaussian Processes, cuantificacion de incertidumbre (UQ) y High-Performance Computing (HPC) con GPU.
* **Meta de Vida:** Emigracion estrategica a Canada (Categoria STEM) a los 31-32 anos mediante un Master de Investigacion financiado o transferencia corporativa (WSP, Hatch, Stantec), utilizando un perfil tecnico de elite para neutralizar un promedio de notas historico (GPA 4.8).

---

## 2. EL PROBLEMA CIENTIFICO (FONDECYT INICIACION 2025)

La investigacion se enmarca en un proyecto de la Agencia Nacional de Investigacion y Desarrollo (ANID) liderado por el Dr. Moris.

### A. El Fenomeno Fisico:

El estudio se centra en la **Determinacion de los Umbrales Criticos de Movimiento Incipiente** en bloques costeros (boulders) ante flujos de alta energia (tipo Tsunami). En las costas de Chile (Bahia Cisnes, Atacama), existen depositos de rocas de hasta 40 toneladas transportadas por paleotsunamis de hace 600 anos.

### B. La Deficiencia de la Ingenieria Tradicional:

Las ecuaciones predictivas actuales (Nandasena, Engel & May) son insuficientes porque:

1. Tratan a las rocas como prismas rectangulares u objetos homogeneos.
2. No consideran la Interaccion Fluido-Estructura (FSI) turbulenta y no lineal.
3. Dependen de coeficientes de friccion y arrastre simplistas que no capturan la geometria irregular.

### C. La Propuesta:

Utilizar el metodo **Smoothed Particle Hydrodynamics (SPH)** acoplado con el motor de dinamica de cuerpos rigidos **ProjectChrono** para simular con alta fidelidad (millones de particulas) el instante exacto en que la hidrodinamica rompe el equilibrio estatico del bloque. Luego, entrenar un **Gaussian Process surrogate** sobre los resultados y ejecutar **Monte Carlo + analisis de Sobol** para cuantificar incertidumbre y sensibilidad parametrica.

---

## 3. LA SOLUCION: LA "REFINERIA DE DATOS" (Pipeline Automatizado)

Se ha construido una infraestructura automatizada en Python que elimina el factor humano y maximiza la precision fisica. Se divide en 4 modulos + fase de UQ:

* **Modulo 1 (Geometry Builder — `src/geometry_builder.py`):** Lee modelos 3D (.stl), calcula volumen, centro de masa e inercia real via `trimesh`, y los inyecta al simulador via XML. Corrige la fisica de raiz (GenCase sobrestima inercia 2-3x a dp gruesos).

* **Modulo 2 (Batch Runner — `src/batch_runner.py`):** Orquesta GenCase + DualSPHysics GPU con timeout adaptativo por dp, cleanup garantizado en `try/finally` (borra .bi4/.vtk), y recovery via `--desde N`. La RTX 5090 no se detiene.

* **Modulo 3 (ETL / Data Cleaner — `src/data_cleaner.py`):** Transforma CSVs de Chrono (separador `;`, sentinel `-3.40282e+38`) en una base SQLite limpia. Detecta automaticamente si la roca se movio mediante criterios cinematicos (desplazamiento del CM + rotacion neta).

* **Modulo 4 (ML Surrogate — `src/ml_surrogate.py`):** Gaussian Process (Matern nu=2.5, scikit-learn) que predice desplazamiento del boulder a partir de parametros de entrada. Soporta N features dinamicamente, genera figuras 2D slices + 1D slices + LOO validation. Exporta `.pkl` para consumo downstream.

* **Fase 5 (UQ — `scripts/run_uq_analysis.py`):** Monte Carlo (10,000 muestras sobre el GP en segundos) + indices de Sobol (algoritmo de Saltelli, implementacion propia). Genera 6 figuras thesis-quality: histograma MC, P(movimiento) vs parametros, tornado Sobol, frontera de probabilidad, bandas CI, convergencia Sobol. Precedente: Salmanidou et al. (2020, Water).

---

## 4. ESTADO ACTUAL (Febrero 2026)

### Completado:

| Hito | Estado | Detalle |
|------|--------|---------|
| Pipeline completo (4 modulos) | FUNCIONAL | Probado end-to-end, deployable en WS |
| Convergencia de malla | ALCANZADA | 7 valores de dp, convergencia en dp=0.003 (delta <5% en 3 metricas). dp=0.004 para produccion |
| Render pipeline | PARCIAL | VTK→PLY completado (21 frames, 26M particulas), Blender pendiente |
| Tesis (3 capitulos) | ESCRITOS | cap3 metodologia, cap4 resultados convergencia, cap5 pipeline |
| Defensa tecnica | PREPARADA | DEFENSA_TECNICA.md (700+ lineas), 28 referencias con DOIs |
| Framework UQ | IMPLEMENTADO | MC + Sobol + 6 figuras, testeado con datos sinteticos |
| Repositorio Git | ACTIVO | github.com/kcortes765/Tesis (4 commits) |

### Resultado Clave — Convergencia (7 dp):

| dp | Desplaz. | Delta% | Rot | F_SPH | Tiempo |
|---|---|---|---|---|---|
| 0.020 | 3.495m | — | 95.8 | 166.4N | 13 min |
| 0.004 | 1.615m | 6.4% | 84.8 | 22.8N | 260 min |
| 0.003 | 1.553m | 3.9% | 90.2 | 22.2N | 812 min |

**Hallazgo cientifico:** La fuerza de contacto NO converge (CV=82%). No sirve como criterio de movimiento incipiente — usar desplazamiento.

### En Ejecucion:

* **Estudio piloto**: 35 casos LHS con 3 parametros (dam_height, boulder_mass, boulder_rot_z) a dp=0.004 en la WS. Tiempo estimado: ~6 dias.
* **Esperando Dr. Moris**: Email enviado 2026-02-20 con preguntas sobre rangos parametricos, geometrias adicionales y criterio de movimiento.

### Proximo:

1. Correr 35 simulaciones piloto en WS (dp=0.004)
2. Entrenar GP con datos reales
3. Monte Carlo + Sobol sobre GP
4. Generar todas las figuras para la tesis
5. Cuando Moris responda: ajustar rangos y repetir con 50 casos definitivos

---

## 5. INFRAESTRUCTURA

* **Estacion de Produccion (WS UCN):** RTX 5090 32GB VRAM. Path: `C:\Users\ProyBloq_Cortes\Desktop\SPH-Convergencia`. Sin git — deploy via ZIP desde GitHub.
* **Estacion de Desarrollo (Laptop):** i7-14650HX + RTX 4060 8GB. Git + gh CLI instalados. Desarrollo, monitoreo, analisis, figuras.
* **Software Core:** DualSPHysics v5.4.3 + ProjectChrono (RigidAlgorithm=3), Python 3.10+, scikit-learn, scipy, trimesh, lxml, matplotlib.
* **Repositorio:** github.com/kcortes765/Tesis (publico)
* **Blender 4.2** en WS para render fotorrealista (pendiente)

---

## 6. ESTRUCTURA DEL PROYECTO

```
Tesis/
├── src/                    # Pipeline principal (4 modulos)
│   ├── geometry_builder.py
│   ├── batch_runner.py
│   ├── data_cleaner.py
│   ├── ml_surrogate.py
│   └── main_orchestrator.py
│
├── scripts/                # Scripts standalone
│   ├── run_production.py       # "Boton rojo" para WS (--pilot --prod)
│   ├── run_uq_analysis.py     # MC + Sobol + figuras UQ
│   ├── run_convergence.py     # Estudio de convergencia
│   ├── figuras_convergencia_7dp_es.py
│   └── [5 scripts mas]
│
├── config/                 # Configuracion
│   ├── param_ranges.json       # Rangos parametricos (3 variables)
│   ├── template_base.xml       # Template DualSPHysics
│   └── dsph_config.json        # Paths y defaults
│
├── data/                   # Resultados
│   ├── figuras_7dp_es/         # Convergencia (10 PNG, espanol)
│   ├── figuras_ml/             # GP surrogate (6 PNG+PDF)
│   ├── figuras_uq/             # UQ analysis (12 PNG+PDF + JSON)
│   ├── results.sqlite          # Base de datos de resultados
│   └── reporte_convergencia_7dp.csv
│
├── tesis/                  # Capitulos de tesis (3 escritos)
├── models/                 # STLs (BLIR3.stl, Canal)
├── DEFENSA_TECNICA.md      # Documento de defensa (28 refs)
├── EDUCACION_TECNICA.md    # Guia tecnica SPH (10 partes)
└── RETOMAR.md              # Punto de entrada para continuidad
```

---

## 7. FORTALEZA CIENTIFICA

### Pipeline con precedente en la literatura:

```
Simulacion SPH (WS, semanas)
  → 35-50 simulaciones GPU (dp=0.004, ~4h cada una)
  → ETL automatico → SQLite
        ↓
GP Surrogate (laptop, minutos)
  → Matern nu=2.5, LOO cross-validation
  → Predice en milisegundos
        ↓
Monte Carlo + Sobol (laptop, segundos)
  → 10,000 muestras → distribucion + CI 95%
  → Indices de Sobol → que parametro domina
  → Frontera de estabilidad probabilistica
```

**Referencia directa:** Salmanidou et al. (2020). "UQ of Landslide Generated Waves Using GP Emulation and Variance-Based Sensitivity Analysis." *Water*, 12(2), 416. Misma metodologia (SPH + GP + Sobol), diferente aplicacion (transporte de boulders con geometria irregular).

### 28 referencias academicas con DOIs, incluyendo:
- Crespo et al. (2015) — DualSPHysics
- Rasmussen & Williams (2006) — Gaussian Processes
- Sobol (2001), Saltelli (2002) — Analisis de sensibilidad
- Nandasena et al. (2011) — Ecuaciones empiricas de comparacion
- Loeppky et al. (2009) — Regla 10d para tamano de muestra GP

---

## 8. OBJETIVOS ESTRATEGICOS DE CARRERA

### A. Nivel Academico (Prestigio Internacional):

Publicar **1-3 Papers en Revistas Q1/Q2** (ej. *Coastal Engineering, Water, SoftwareX*). El pipeline SPH→GP→UQ→Sobol es publicable en al menos 2 papers: uno metodologico (pipeline + convergencia) y uno de resultados (fronteras de estabilidad probabilisticas).

### B. Nivel Laboral (Empleabilidad de Elite):

Postular a practicas en BHP, SQM, WSP, Hatch como **Computational Engineer**. El ROI: *"Mi codigo reduce 6 meses de modelacion manual a un fin de semana de computo desatendido."*

### C. Nivel Migratorio (Objetivo: Canada):

La "Puerta Trasera" academica: contactar PIs en Canada (UBC, UCalgary, UVic) ofreciendo codigo productivo y capacidad de generar papers rapidos. Paper Q1 [Under Review] = pasaporte para negociar con profesores.

### D. Nivel de Negocios (Consultora ConTech):

El pipeline es el MVP de una consultora tecnologica. Automatizar pre/post-proceso de ingenieria civil es un servicio de alto margen en el sector minero y portuario.

---

## 9. CONCLUSION

La tesis opera al **90% de capacidad**. Los 4 modulos del pipeline estan funcionales, la convergencia esta demostrada, el framework UQ esta implementado y testeado, y 3 capitulos de tesis estan escritos. El cuello de botella es la respuesta del Dr. Moris con rangos parametricos definitivos. Mientras tanto, el estudio piloto de 35 casos esta listo para correr en la WS.

Al final de este proyecto, Kevin Cortes no sera un graduado buscando empleo — sera un investigador con codigo productivo, datos publicables y una metodologia que el mercado internacional reconoce.

**Veredicto:** Pipeline blindado. Convergencia demostrada. UQ listo. Solo falta tiempo de computo.
