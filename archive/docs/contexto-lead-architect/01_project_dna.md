# PROJECT DNA & MASTER VISION

## 1. IDENTIFICACION Y PERFIL DEL LIDER DEL PROYECTO

* **Investigador Principal:** Kevin Cortes Hernandez (Sebastian Cortes para fines de branding internacional).
* **Institucion:** Universidad Catolica del Norte (UCN), Antofagasta, Chile.
* **Situacion Academica:** Alumno de 5to ano de Ingenieria Civil, cursando Pre-Capstone y Capstone Project (2026).
* **Mentor Academico:** Dr. Joaquin Moris Barra (Secretario de Investigacion FaCIC / Investigador Principal Fondecyt Iniciacion).
* **Perfil Tecnico:** Computational Civil Engineer & Data Specialist. Experto en automatizacion de procesos de ingenieria, MLOps aplicados a la fisica y High-Performance Computing (HPC).
* **Meta de Vida:** Emigracion estrategica a Canada (Categoria STEM) a los 31-32 anos mediante un Master de Investigacion financiado o transferencia corporativa (WSP, Hatch, Stantec), utilizando un perfil tecnico de elite para neutralizar un promedio de notas historico (GPA 4.8).

---

## 2. EL PROBLEMA CIENTIFICO (FONDECYT INICIACION 2025)

La investigacion se enmarca en un proyecto de la Agencia Nacional de Investigacion y Desarrollo (ANID) liderado por el Dr. Moris.

### A. El Fenomeno Fisico:

El estudio se centra en la **Determinacion de los Umbrales Criticos de Movimiento Incipiente** en bloques costeros (cantos rodados o boulders) ante flujos de alta energia (tipo Tsunami). En las costas de Chile (especificamente Bahia Cisnes, Atacama), existen depositos de rocas de hasta 40 toneladas transportadas por paleotsunamis de hace 600 anos.

### B. La Deficiencia de la Ingenieria Tradicional:

Las ecuaciones predictivas actuales (Nandasena, Engel & May) son insuficientes porque:

1. Tratan a las rocas como prismas rectangulares u objetos homogeneos.
2. No consideran la Interaccion Fluido-Estructura (FSI) turbulenta y no lineal.
3. Dependen de coeficientes de friccion y arrastre simplistas que no capturan la realidad de la geometria irregular.

### C. La Propuesta Disruptiva:

Utilizar el metodo **Smoothed Particle Hydrodynamics (SPH)** acoplado con el motor de dinamica de cuerpos rigidos **ProjectChrono** para simular con alta fidelidad (millones de particulas) el instante exacto en que la hidrodinamica rompe el equilibrio estatico del bloque.

---

## 3. LA INNOVACION TECNOLOGICA (EL VALOR AGREGADO DE KEVIN)

Este proyecto ha dejado de ser una tesis descriptiva para convertirse en el desarrollo de una **"Refineria de Datos Hidraulicos"**. La innovacion core no es el uso del software, sino la **arquitectura de automatizacion**.

### El Salto de Eficiencia:

El flujo de trabajo heredado del laboratorio (investigador Diego) era manual y artesanal: edicion de XML en Notepad++, calculos de volumen en AutoCAD, inspeccion visual de videos en ParaView y gestion de datos en Excel con errores de formato regional.

**Kevin Cortes ha implementado un sistema "Headless" que:**

1. **Elimina el Error de Fisica:** Calcula mediante Python (`trimesh`) el Tensor de Inercia y Volumen real de mallas 3D (.stl), inyectando estos valores directamente en el XML para anular el error de voxelizacion de DualSPHysics.
2. **Orquestacion Masiva (LHS):** Utiliza *Latin Hypercube Sampling* para generar y ejecutar campanas de cientos de simulaciones en serie en una GPU RTX 5090 sin intervencion humana.
3. **Pipeline ETL Automatizado:** Un motor en Python que limpia Gigabytes de archivos binarios, extrae cinematica precisa del bloque, evalua matematicamente el fallo y almacena todo en una base de datos SQLite.
4. **Surrogate Modeling (ML):** El uso de los datos generados para entrenar una IA de Procesos Gaussianos que prediga la estabilidad en milisegundos, ahorrando semanas de computo.

---

## 4. INFRAESTRUCTURA DE HARDWARE Y SOFTWARE

* **Estacion de Produccion:** Workstation RTX 5090 (32GB VRAM). Esta maquina es la "trituradora de datos". Se encarga de las simulaciones de alta fidelidad (dp < 0.005m / 50M+ particulas).
* **Estacion de Desarrollo:** Laptop RTX 4060 (8GB VRAM). Se usa para prototipado, depuracion de codigo Python y simulaciones de baja resolucion (Smoke Tests).
* **Software Core:** DualSPHysics v5.4.3 (Motor SPH), ProjectChrono (Motor de Colisiones), Python 3.10+ (Orquestador), Claude Code (Asistente de Desarrollo CLI).

---

## 5. OBJETIVOS ESTRATEGICOS DE CARRERA (THE BIG PICTURE)

### A. Nivel Academico (Prestigio Internacional):

Publicar **de 1 a 3 Papers en Revistas Q1/Q2** (ej. *Coastal Engineering, Water, SoftwareX*). Estos papers son el "Escudo" contra el promedio 4.8. El objetivo es que el mundo academico vea a un investigador experto en HPC y MLOps, no a un alumno con notas bajas. El estado *[Under Review]* en una revista Q1 es el pasaporte inmediato para negociar con profesores en Canada.

### B. Nivel Laboral (Empleabilidad de Elite):

Postular a practicas y trabajos en la "Primera Division" minera e industrial de Chile (BHP, SQM, WSP, Hatch) posicionandose como **Computational Engineer**. El discurso de venta es el Retorno de Inversion (ROI): *"Mi codigo reduce 6 meses de modelacion manual a un fin de semana de computo desatendido"*.

### C. Nivel Migratorio (Objetivo: Canada):

Utilizar la "Puerta Trasera" academica. Contactar directamente a Profesores Investigadores (PIs) en Canada (UBC, UCalgary, UVic) ofreciendo el codigo y la capacidad de producir papers rapidos. El profesor, financiado por *Grants* del gobierno, actuara como patrocinador ante el comite de admisiones, ignorando el GPA historico.

### D. Nivel de Negocios (Consultora ConTech):

Este pipeline es el MVP (Producto Minimo Viable) de una futura consultora tecnologica. La capacidad de automatizar pre y post-proceso de ingenieria civil es un servicio de altisimo margen de ganancia en el sector privado minero y portuario.

---

## 6. MENTALIDAD OPERATIVA (RULES OF ENGAGEMENT)

1. **Automatizar es el Mandato:** Si una tarea toma mas de 5 minutos y se repite, debe ser un script de Python.
2. **Rigor Fisico:** El codigo no sirve si la fisica es falsa. Cada parametro (dp, Visco, FtPause) debe ser justificado academicamente.
3. **Gestion de Recursos:** La RTX 5090 no debe estar ociosa. El tiempo de computo es el activo mas caro del proyecto.
4. **Comunicacion Senior:** Kevin no es un alumno pidiendo permiso; es un arquitecto de soluciones reportando avances y solicitando definiciones de borde (boundary conditions) al Dr. Moris.
