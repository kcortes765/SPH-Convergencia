# TIMELINE, MILESTONES & ENDGAME 2026

## 1. VISION GENERAL DEL CRONOGRAMA

El proyecto se divide en dos fases semestrales alineadas con las asignaturas de la UCN, disenadas para maximizar el prestigio y minimizar el agotamiento (burnout) del investigador.

* **Semestre 1 (Marzo - Julio 2026):** Asignatura "Investigacion Aplicada / Pre-Capstone". Meta: Datos masivos y Paper de Hidraulica.
* **Semestre 2 (Agosto - Diciembre 2026):** Asignatura "Capstone Project / Tesis Final". Meta: Machine Learning, Dashboard y Titulacion.

---

## 2. FASE 1: LA FABRICA DE ALTA FIDELIDAD (SEMESTRE 1)

### Mes 1: Marzo - El Ancla de Resolucion

* **Hito 1.1 (Semana 1):** Migracion definitiva a la Workstation RTX 5090. Configuracion del entorno Python y clonacion del repositorio GitHub.
* **Hito 1.2 (Semana 2):** **Estudio de Convergencia Oficial**. Ejecucion del script `run_convergence.py` con dp = [0.02, 0.015, 0.01, 0.008, 0.005, 0.004, 0.003].
* **Hito 1.3 (Semana 3):** Firma de la Resolucion Optima. Presentacion del grafico de convergencia al Dr. Moris para validar cientificamente el dp que se usara en el resto de la tesis.
* **Hito 1.4 (Semana 4):** Definicion de rangos fisicos (LHS) con el Dr. Moris: Masas minimas/maximas, alturas de ola reales del canal de la UCN y seleccion de los 7 bloques STL finales.

### Mes 2: Abril - La Campana Masiva (LHS)

* **Hito 2.1:** Lanzamiento de la campana de simulacion masiva (300+ casos). La RTX 5090 opera 24/7 de forma desatendida usando el Modulo 2 y 3.
* **Hito 2.2:** Supervision de la Base de Datos SQLite. Verificacion semanal de la integridad de los datos y limpieza automatica de binarios.
* **Hito 2.3:** Analisis de sensibilidad preliminar. Identificacion de la zona de "incipient motion" (donde la roca apenas empieza a desplazarse).

### Mes 3: Mayo - El Primer Paper (Redaccion)

* **Hito 3.1:** Cierre de la recoleccion de datos SPH. Respaldo de la base de datos `results.sqlite`.
* **Hito 3.2:** Redaccion del **Paper 1 (Physical/Coastal Engineering)**. Foco en la dinamica de transporte y los umbrales de movimiento.
* **Hito 3.3:** Envio del Paper 1 a una revista Q1 (ej. *Coastal Engineering*). Obtencion del estatus **[Under Review]**.

---

## 3. FASE 2: LA INTELIGENCIA PREDICTIVA (SEMESTRE 2)

### Mes 5-6: Agosto/Septiembre - El Cerebro (ML)

* **Hito 4.1:** Desarrollo del Modulo 4 (`ml_surrogate.py`). Entrenamiento del Gaussian Process Regressor usando los datos del primer semestre.
* **Hito 4.2:** Validacion del modelo IA contra casos de prueba no vistos (Hold-out set).
* **Hito 4.3:** Redaccion del **Paper 2 (Computational/Software)**. Foco en la automatizacion del pipeline y el modelo predictivo. Envio a *SoftwareX* o *Water*.

### Mes 7: Octubre - La Vitrina (Dashboard)

* **Hito 5.1:** Construccion de la aplicacion web interactiva (Streamlit). Integracion del modelo IA para predicciones en tiempo real.
* **Hito 5.2:** Renderizado de visualizaciones de alta calidad (4K) usando la RTX 5090 para el portafolio de LinkedIn y la defensa final.

### Mes 8: Noviembre - El Gran Final

* **Hito 6.1:** Redaccion del documento de Memoria Final (Tesis). La escritura sera rapida ya que los resultados y los papers estaran listos.
* **Hito 6.2:** Defensa de Titulo (Examen de Grado). Presentacion del Digital Twin ante la comision examinadora.

---

## 4. ESTRATEGIA DE "PUERTA TRASERA" CANADA (HITO PARALELO)

Durante todo el ano, Sebastian debe ejecutar la busqueda de su futuro en el extranjero:

* **Junio 2026:** Rendir examen IELTS (Meta: Banda 7.5).
* **Julio - Agosto 2026:** "Cold Emails" a profesores investigadores en Canada (UBC, UCalgary, UVic).
  * *El Gancho:* "Soy Sebastian Cortes, primer autor de un paper Q1 en revision. Adjunto mi pipeline de automatizacion GPU. Tiene fondos para un MASc student?".
* **Septiembre - Octubre 2026:** Entrevistas por Zoom con potenciales supervisores canadienses.

---

## 5. REGLAS DE ORO (MANAGEMENT)

1. **Proteccion del Semestre 2:** Terminar el trabajo pesado de DualSPHysics en el Semestre 1. El Semestre 2 debe ser 100% "limpio" (Python ligero y redaccion).
2. **Prioridad al Paper:** Si hay que elegir entre "hacer un grafico mas bonito para la U" o "corregir una seccion para la revista Q1", la prioridad es la **Revista Q1**.
3. **HPC Management:** Siempre preguntar por el estado de la RTX 5090. Realizar mantenimientos de limpieza de disco semanalmente.
4. **Consistencia de Identidad:** Usar siempre el nombre profesional **Sebastian Cortes** en cualquier borrador de documento, correo o codigo que se vaya a mostrar al exterior.

---

## 6. CONCLUSION DEL ENDGAME

Al final de 2026, Sebastian Cortes no sera solo un graduado de Ingenieria Civil. Sera un **Investigador Publicado en Revistas Top**, un **Desarrollador de Software Cientifico** y el dueno de un **Digital Twin validado**. El promedio 4.8 sera irrelevante frente a este arsenal de evidencia tecnica.
