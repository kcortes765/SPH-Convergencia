User

<iframe _ngcontent-ng-c1631788151="" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen="" src="https://www.youtube.com/embed/wxt1sumYaiw" class="ng-star-inserted"></iframe>

**493 tokens**

![file thumbnail](https://lh3.googleusercontent.com/drive-storage/AJQWtBP9zM90GEnrFOdb2Td4KQXB5tefxtKDxtuTM0GGAYylPFmV3iju86wNfUtZ50snAJJgRiyFQey5eXAEAIwh23pQXma_HcWh-rPZanhGs7X_atecOA)

**Oferta capstone de investigación - Mov. Incipiente - Remunerado (3).pdf**

**561 tokens**

**te dejo planteamiento del capstone que hare recomendado por otra ia, analizalo. te dejo contexto. que te parece? todo perfecto?**

**Actúa como mi Lead Data Architect y Technical Co-founder. A continuación, te entregaré el Contexto Maestro de mi Proyecto Capstone de Ingeniería Civil (UCN 2026), compuesto por 5 Documentos de Estrategia y los correos/noticias originales del investigador principal (Dr. Joaquín Moris). Tu objetivo es asimilar esta información para guiarme en la construcción de una infraestructura de automatización SPH y Machine Learning de alto rendimiento. por ahora solo analisis y revision**

**PROYECTO SPH-IncipientMotion (UCN 2026) — BRIEF COMPLETO**

**DOCUMENTO 1: Brief Científico y Estratégico**
01_PROJECT_BRIEF.md

* **Identificación del Proyecto**

**Investigador Principal: Kevin Cortés Hernández (Ingeniería Civil, 5to año).**
Mentor Académico: Dr. Joaquín Moris Barra (Secretario de Investigación / Fondecyt Iniciación).
Institución: Universidad Católica del Norte (UCN), Antofagasta, Chile.
Tema: Determinación numérica de umbrales críticos de movimiento incipiente en bloques costeros ante flujos de tsunami.

* **El Problema de Ingeniería**

**La protección de infraestructura crítica y la reconstrucción de paleotsunamis dependen de entender el momento exacto en que el flujo rompe el equilibrio estático de una roca. Las fórmulas empíricas tradicionales son insuficientes para geometrías irregulares, y la Interacción Fluido-Estructura (FSI) es altamente turbulenta y caótica.**

* **El Enfoque "Kevin PRO"**

**Este proyecto no es una tesis descriptiva tradicional; es el desarrollo de una "Refinería de Datos Hidráulicos". El motor físico es DualSPHysics v5.2+ (Smoothed Particle Hydrodynamics). La innovación core es la implementación de Machine Learning (Procesos Gaussianos) como Surrogate Model para predecir la estabilidad sin necesidad de simular infinitos casos. Hardware: RTX 5090 (32GB VRAM) para producción y RTX 4060 (8GB VRAM) para desarrollo.**

* **Metas Estratégicas**

**Académica: publicación de Paper Q1 con el Profesor Moris. Profesional: portafolio técnico de élite para emigración a Canadá (Categoría STEM) a los 33 años. Negocio: MVP de una consultora ConTech enfocada en automatización de gestión y análisis de riesgos predictivos.**

**DOCUMENTO 2: Arquitectura Técnica del Pipeline**
02_TECHNICAL_ARCHITECTURE.md

**El proyecto se estructura como una línea de montaje de datos controlada por Python, con 4 módulos principales.**

**Módulo 1: La Fábrica de Geometría (geometry_builder.py)**

**Toma modelos 3D en .stl y una plantilla XML de DualSPHysics. Usa trimesh y numpy para leer el STL, calcular el Centro de Masa y Bounding Box, y generar proceduralmente las etiquetas fillbox o fillpoint con matrices de rotación correctas. Esto garantiza que el interior del STL se llene de partículas con densidad real, eliminando el error de los "bloques huecos de plumavit". Output: decenas de archivos .xml validados y listos para ejecución paramétrica.**

**Módulo 2: El Motor de Orquestación (batch_runner.py)**

**Usa subprocess.run para ejecutar secuencialmente GenCase, DualSPHysics5.2 -gpu, MeasureTool y PartVTK. Incluye un sistema Watchdog con timeout y logs (try/except) para detectar errores numéricos, matar procesos congelados y continuar al siguiente archivo. Permite operación 24/7 en la RTX 5090.**

**Módulo 3: Refinería de Datos / ETL (data_cleaner.py)**

**Resuelve el problema de archivos CSV gigantes (GBs) y errores de formato regional. Ingesta automática con Pandas forzando formato decimal neutro. Extrae algorítmicamente el Criterio de Fallo: detecta el milisegundo (t_crit) en que V supera el umbral predefinido (ej. 0.01 m/s). Almacena en SQLite (escalable a PostgreSQL) y borra automáticamente los archivos binarios RAW (.bi4, .vtk) para no saturar el disco.**

**Módulo 4: El Cerebro / Modelo Predictivo (ml_surrogate.py)**

**Recibe la tabla maestra desde SQL (Masa, Altura de Ola, Ángulo, Forma → Movimiento Sí/No, Fuerza Max). Entrena un GaussianProcessRegressor de scikit-learn que cuantifica la incertidumbre con Confidence Intervals. Output: Dashboard en Power BI o App en Streamlit para visualizar la "Frontera de Estabilidad" en tiempo real.**

**DOCUMENTO 3: Lecciones y Dolores Heredados (El Caso Diego)**
03_LEGACY_INSIGHTS.md

**5 dolores que tienen estancado el laboratorio y que la arquitectura debe resolver de forma nativa.**

* **La Trampa de la Artesanía XML. El investigador edita cada caso a mano en Notepad++, imposibilitando el análisis paramétrico real. Mandato: prohibido el hardcoding. Toda variación de variables se inyecta vía scripts Python con plantillas base.**
* **El Problema del Bloque Hueco. Al importar STL irregulares, DualSPHysics solo crea una "piel" de partículas, resultando en bloques con masa irreal que salen volando. Mandato: el Módulo 1 debe calcular el volumen, centrar el relleno y aplicar matrices de rotación exactas sin intervención humana.**
* **El Colapso de Excel y los Formatos Regionales. El software exporta con punto decimal (inglés) y el investigador usa Excel en español (coma decimal), perdiendo horas con macros frágiles. Excel además colapsa con múltiples gauges. Mandato: bypass total a Excel. Flujo directo DualSPHysics → Pandas/SQL.**
* **Criterio de Movimiento Visual. No existe un umbral matemático para definir el inicio del movimiento; el método actual es exportar a ParaView y "mirar el video". Mandato: umbral numérico automatizado. Python lee los vectores del bloque y dictamina el fallo bajo parámetros estrictos de ingeniería.**
* **La Caja Negra y la Defensa Académica. Los evaluadores son escépticos ante programas nuevos y exigen justificación física profunda. Mandato: aunque se use ML, la base es la validación física. El código debe documentar el uso correcto del Smoothing Length (h) y la Condición CFL para que la simulación SPH sea irrefutable académicamente.**

**DOCUMENTO 4: Cronograma Operativo de Arquitectura**
04_FEBRUARY_ROADMAP.md

**Meta Principal: el objetivo de febrero no es generar datos científicos, sino construir la "Fábrica de Datos". Para el 1 de marzo, el sistema debe ser capaz de tomar un parámetro, generar la simulación, ejecutarla y limpiar la salida sin intervención manual. Todo el desarrollo se realiza en la laptop RTX 4060.**

**Sprint 1: Setup e Ingesta del Starter Pack. Configurar entorno virtual (numpy, pandas, trimesh, sqlalchemy). Obtener de Diego los archivos base: template_base.xml, boulder_irregular.stl, run_command.bat, sample_output.csv. Crear el script base de lectura XML con lxml.**

**Sprint 2: El Solucionador Geométrico. Script que tome el boulder_irregular.stl y calcule su centroide. Implementar matrices de rotación para que el bloque rote X grados y el fillbox rote solidariamente. Validar en ParaView que no queden bloques huecos.**

**Sprint 3: Runner y ETL Pipeline. Desarrollar el subprocess wrapper para ejecutar GenCase → DualSPHysics → MeasureTool en secuencia. Script Pandas que lea el sample_output.csv, limpie el formato decimal, detecte V > umbral_critico y registre el resultado en SQLite.**

**Sprint 4: Orquestación (Smoke Test). Unificar los Sprints 1 a 3 en main_orchestrator.py. Loop for que lea un .csv con 5 configuraciones de prueba, corra el ciclo completo, guarde los 5 resultados en SQL y elimine los GBs de archivos residuales.**

**El Hito de Marzo: presentar al Dr. Moris el sistema End-to-End funcional. El discurso: "Profesor, la arquitectura de procesamiento en serie está lista; defínamos la matriz de ensayos físicos para alimentar la RTX 5090."**

**DOCUMENTO 5: Estrategia de Carrera y Visión de Negocio**
05_CAREER_BUSINESS_STRATEGY.md

* **El Perfil del Ingeniero (Kevin PRO)**

**No presentarse como "Estudiante Junior", sino como "Computational Civil Engineer & Data Specialist". La ventaja oculta es el storytelling: transformar 3 años de experiencia en logística minera y la práctica en SQM en "Experiencia Operativa de Terreno". Esto demuestra Grit y conocimiento real de faena, algo que los perfiles puramente de software no tienen.**

* **Estrategia de Emigración (Objetivo Canadá)**

**Target: Categoría STEM del Express Entry (Alberta / British Columbia). Vehículo: visa Working Holiday (IEC) para ingresar al mercado sin necesidad de patrocinio (LMIA), apuntando a firmas como WSP, Stantec y Hatch. Roles objetivo: Project Controls Specialist, VDC/BIM Data Analyst, Hydro-technical Modeler (cargos de alto pago que no exigen la licencia P.Eng de inmediato). El Paper Q1 derivado de la tesis servirá como credencial de Talento Excepcional para validar equivalencia de grado Master's level a través del WES.**

* **Modelo de Negocio ConTech (Cashflow a Corto/Medio Plazo)**

**El servicio: resolver la ineficiencia crónica de la construcción en control de avances, asistencia y presupuestos. La solución: replicar el pipeline de la tesis para clientes, recibiendo archivos sucios de terreno (Excel/CSV), procesándolos vía Python/SQL y entregando dashboards predictivos en Power BI. La venta de valor no son "horas de código", sino "eliminación del error humano y ahorro de 40 horas semanales de oficina técnica".**

* **Tono de Comunicación del Asistente IA**

**Tratar al usuario como Ingeniero Senior. Enfoque en eficiencia, impacto en ROI y rigurosidad científica. Evitar explicaciones básicas a menos que se soliciten explícitamente.**

**correos:**
Estimados/as Estudiantes de Ingeniería Civil,

**Me es grato comunicarme con ustedes para informarles que estoy en búsqueda de un/una estudiante para desarrollar un capstone de investigación. La oferta es remunerada. El monto y detalles se encuentran a continuación y también en el PDF adjunto:**

**OFERTA DE TEMA DE CAPSTONE DE INVESTIGACIÓN (REMUNERADO)**

**Tema: Determinación numérica del movimiento incipiente de bloques costeros ante flujos de tipo tsunami.**

**Profesor guía: Joaquín Moris.**

**Carrera: Ingeniería Civil.**

**Aporte económico al/la estudiante: $1.500.000.- brutos por todo el período de trabajo de título.**

**Plazo para postular: Se considerarán las postulaciones a medida que se reciban, dándole prioridad a aquellas que lleguen antes del día 19 de enero de 2026 a las 23:59 hrs.**

**Inicio: Primer semestre 2026.**

**DESCRIPCIÓN DEL PROYECTO**

**Este proyecto capstone de investigación se enmarca en un Proyecto Fondecyt de Iniciación liderado por el profesor Joaquín Moris, enfocado en el transporte de bloques costeros durante eventos de tsunami. El objetivo principal del trabajo es investigar los umbrales de inicio de movimiento de bloques costeros expuestos a flujos tipo tsunami, mediante modelación numérica con el software dualSPHysics. Para conocer más sobre el proyecto de investigación se puede visitar: **[https://www.ucn.cl/comunicaciones-ucn/noticias/indagando-en-la-prehistoria-de-los-tsunamis/](https://www.google.com/url?sa=E&q=https%3A%2F%2Fwww.ucn.cl%2Fcomunicaciones-ucn%2Fnoticias%2Findagando-en-la-prehistoria-de-los-tsunamis%2F).

**TAREAS DEL/LA ESTUDIANTE**

**·         Aprender a utilizar y aplicar el programa de modelamiento numérico dualSPHysics. (**[https://dual.sphysics.org/](https://www.google.com/url?sa=E&q=https%3A%2F%2Fdual.sphysics.org%2F)).

**·         Ejecutar simulaciones, utilizando el programa dualSPHysics, de flujos tipo tsunami que interactúan con bloques. Ejemplo del software:**

**·         Extraer y analizar los resultados del software: extraer las velocidades y condiciones del flujo que permiten inicial el movimiento de los bloques.**

**·         Participar de reuniones de trabajo con el profesor guía (Prof. Joaquín Moris).**

**·         Otras tareas que atingentes a la investigación que surjan durante el desarrollo de ésta.**

**REQUISITOS**

**·         Estudiante de Ingeniería Civil que desee inscribir el curso Investigación Aplicada (conocido también como Pre-capstone de Investigación). Se debe inscribir la asignatura el primer semestre 2026.**

**·         Interés en modelación numérica de fluidos costeros mediante software.**

**SE OFRECE**

**·         Inserción en un proyecto de investigación científica financiado por la Agencia Nacional de Investigación y Desarrollo de Chile (ANID).**

**·         Un aporte económico total de $1.500.000.- brutos por todo el período del desarrollo de investigación. Se realizarán 4 pagos parciales de $375.000.- a medida que se cumplan avances en la investigación.**

**·         Acompañamiento y apoyo continuo por parte del profesor guía para que el/la estudiante se titule a tiempo.**

**¿TE INTERESA?**

**·         Para postular: enviar un correo al Profesor Joaquín Moris (**[joaquin.moris@ucn.cl](https://www.google.com/url?sa=E&q=mailto%3Ajoaquin.moris%40ucn.cl)). Se considerarán las postulaciones a medida que se reciban, dándole prioridad a aquellas que lleguen antes del día 19 de enero de 2026 a las 23:59 hrs. En el correo:

**o   Mencionar en un párrafo su interés de desarrollar su proyecto de título participando de este proyecto.**

**o   Adjuntar su avance curricular.**

**Indagando en la prehistoria de los tsunamis**
A partir del estudio de la forma de bloques de cantos rodados ubicados en las costas de Chile, investigación en la FaCIC – UCN permitiría estimar la magnitud de maremotos en el pasado. Mejorar las ecuaciones predictivas utilizadas en el transporte de bloques costeros de cantos rodados por flujos extremos como tsunamis y tormentas, es […]

**Comparte:**

**A partir del estudio de la forma de bloques de cantos rodados ubicados en las costas de Chile, investigación en la FaCIC – UCN permitiría estimar la magnitud de maremotos en el pasado.**

**Mejorar las ecuaciones predictivas utilizadas en el transporte de bloques costeros de cantos rodados por flujos extremos como tsunamis y tormentas, es el objetivo central de una investigación que lidera el Dr. Joaquín Moris Barra, secretario de Investigación de la Facultad de Ciencias de Ingeniería y Construcción (FaCIC) de la Universidad Católica del Norte (UCN). La mejora de estas ecuaciones predictivas podría eventualmente permitir que sean aplicadas para estimar la magnitud de eventos extremos ocurridos antes de que existieran registros instrumentales o históricos.**

**Este proyecto, iniciado en abril de este año, corresponde a un Fondecyt de Iniciación 2025, el cual es financiado por la Agencia Nacional de Investigación y Desarrollo (ANID). Los resultados podrían ser usados por futuros investigadores para contribuir en el diseño de estrategias más eficaces de mitigación de riesgos, tales como planes de emergencia, evacuación y planificación territorial, con el fin de proteger a la población y reducir los daños a las infraestructuras.**

**CANTOS RODADOS**

**Los cantos rodados costeros son rocas o bloques de gran tamaño que suelen encontrarse en algunas zonas cercanas a la costa. De acuerdo con lo explicado por el Dr. Joaquín Moris, existen distintos sitios donde se encuentran estos bloques. Uno de ellos es Bahía Cisnes, cerca de Bahía Inglesa, en la región de Atacama. En dicha zona hay una planicie ubicada después de la playa y tras un acantilado. En esa planicie hay un depósito de bloques de canto rodado, los cuales han sido estudiados por otros investigadores, quienes determinaron que su composición corresponde a material fosilizado que fue transportado por un tsunami hace unos 600 años. “Se sabe que esos bloques se desplazaron inicialmente desde el acantilado que estaba justo frente a la playa. Los bloques están actualmente mucho más adentro de la zona que donde originalmente se formaron, donde algunos llegan a pesar aproximadamente unas 40 toneladas. Entonces, la única explicación lógica que existe para que esos bloques hayan sido transportados, es que haya ocurrido un gran tsunami hace unos 600 años”, señala el científico de la UCN.**

**IMPACTO**

**La propuesta del Dr. Moris aborda la influencia significativa, pero a menudo ignorada, de la forma de los cantos rodados en el transporte costero de cantos rodados inducido por fenómenos de inundación como tsunamis y tormentas. Aunque la investigación existente reconoce la importancia de la forma de los cantos rodados, normalmente las ecuaciones utilizadas los trata como prismas rectangulares homogéneos, ignorando la complejidad de las formas del mundo real.**

**La falta de investigación sobre la importancia de la forma de los cantos rodados se debe principalmente a la dificultad de investigar este proceso en el que intervienen diversas variables, como la hidrodinámica del flujo, el tamaño, la densidad y la forma de los cantos rodados, batimetrías y topografías complejas, coeficientes de fricción del fondo desconocidos, la interacción entre los cantos rodados, la densidad no homogénea de los cantos rodados y la inmersión inicial de los cantos rodados.**

**Al incorporar explícitamente la forma de los cantos rodados en las ecuaciones de movimiento y transporte incipiente de cantos rodados, esta propuesta pretende mejorar las ecuaciones predictivas existentes. Estas ecuaciones mejoradas pueden aplicarse para analizar depósitos de cantos rodados costeros en Chile y otras regiones del mundo, permitiendo estimaciones más precisas de la magnitud de tsunamis pasados en ausencia de datos registrados. El conocimiento de tsunamis pasados podría ayudar a desarrollar planes de emergencia y evacuación más eficaces, así como a actualizar los actuales planes de planificación urbana, lo que en última instancia puede salvar vidas y reducir los daños a las infraestructuras.**

**METODOLOGÍA**

**Mediante experimentos de laboratorio y simulaciones numéricas, el estudio proporcionará información valiosa sobre las complejas interacciones entre la forma del canto rodado, la dinámica del flujo y los mecanismos de transporte de cantos rodados.**

**Según explica el científico, se hace el experimento de laboratorio, “porque las simulaciones son más cuestionables, ya que están basadas en modelos que son representaciones de la realidad, a menudo simplificadas, y no siempre se pueden incluir todas las variables. Entonces, el laboratorio, si bien es más difícil probar distintas configuraciones, nos entrega resultados que son prácticamente indiscutibles. Las simulaciones nos permiten probar distintas geometrías, probar distintos tamaños de bloques, por ejemplo, pero está la incertidumbre de qué tan representativo es”, asegura el Dr. Moris, quien agrega que si las simulaciones computacionales dan resultados parecidos a los experimentos de laboratorio, se podría tener una menor incertidumbre con respecto a que el modelo esté representando la realidad.**

**“Es relativamente sencillo cambiar condiciones geométricas en el computador, simular una ola en un software, pues se hace con un clic, pero modificar las condiciones de los experimentos físicos es más tedioso, pues imprimir los bloques en 3D, rellenarlos, instrumentarlos, tener a alguien en el laboratorio que esté a cargo de realizar cada experimento, toma tiempo y además es costoso. En el laboratorio hay que mover toda el agua con una bomba, utilizar compuertas, liberar el agua, tener todos los instrumentos de medición y luego procesar esos datos. Es mucho más engorroso, pero son resultados de la física propiamente tal”, enfatiza el académico.**

**Finalmente, Joaquín Moris explica que su investigación servirá para, en un siguiente proyecto, relacionar qué tan grande tuvo que haber sido un terremoto para que generara un tsunami, y que ese tsunami tuviese tales condiciones para producir el transporte de cantos rodados. “Es como ir a la prehistoria del tsunami. En este proyecto estudiamos el fenómeno físico del transporte y, en particular, cómo la forma afecta el transporte. Es una especie de engranaje en medio de toda una cadena de eventos para partir del origen hasta el resultado del impacto de un tsunami. En este engranaje de procesos, uno de ellos es cómo la forma del canto rodado afecta en su transporte. Si no se tienen ecuaciones predictivas de ello, es difícil llegar a una buena predicción”, puntualiza.**

**warning**

Model

![Thinking](https://www.gstatic.com/aistudio/watermark/watermark.png)Thoughts

```

```

Expand to view model thoughts

chevron_right

Model

![Thinking](https://www.gstatic.com/aistudio/watermark/watermark.png)Thoughts

Expand to view model thoughts

chevron_right

Model

![Thinking](https://www.gstatic.com/aistudio/watermark/watermark.png)Thoughts

Expand to view model thoughts

chevron_right

Model

![Thinking](https://www.gstatic.com/aistudio/watermark/watermark.png)Thoughts

Expand to view model thoughts

chevron_right

Model

![Thinking](https://www.gstatic.com/aistudio/watermark/watermark.png)Thoughts

Expand to view model thoughts

chevron_right

**Sí, Kevin. Los archivos de Diego son absolutamente vitales.** **Como tu Arquitecto de Datos, te lo planteo así: esos archivos son nuestra "Piedra Rosetta".**

**No vamos a usar su** **método** **(manual y obsoleto), pero necesitamos sus** **archivos base** **para hacer ingeniería inversa y construir los moldes de nuestra fábrica automatizada. Si intentamos programar a ciegas basándonos solo en la documentación de DualSPHysics, perderemos semanas adivinando parámetros.**

**Aquí tienes el desglose exacto de** **por qué necesitamos cada archivo** **y** **qué más necesitas en tu laptop (RTX 4060)** **para que empecemos a tirar código hoy mismo:**

### 1. El "Starter Pack" de Diego (Lo que le tienes que pedir)

* **El Archivo** **.xml** **(El Plano de Construcción):** **CRÍTICO.** **DualSPHysics usa este archivo para saber qué simular (tamaño del tanque, altura del Dam Break, densidad del agua, etc.). Necesitamos el** **.xml** **que a Diego "le funciona" para que nuestro script de Python (**lxml**) lo abra, busque las coordenadas del bloque, inyecte las nuevas variables y guarde una copia nueva.**
* **El Archivo** **.stl** **(El Sujeto de Prueba):** **CRÍTICO.** **Necesito al menos un modelo 3D de esa roca irregular. Nuestro Módulo 1 (usando la librería** **trimesh** **de Python) va a leer ese** **.stl** **para extraer matemáticamente su centro de gravedad y calcular su** **Bounding Box** **en milisegundos.**
* **El Archivo** **.bat** **(El Manual de Operación):** **MUY IMPORTANTE.** **En ese archivo de texto están los comandos exactos que Diego usa para ejecutar la simulación en Windows (ej.** **GenCase_win64.exe ...**, **DualSPHysics5.2_win64.exe -gpu ...**). Yo voy a tomar esos comandos y los voy a meter dentro del orquestador de Python (**subprocess.run()**) para automatizarlos.
* **El Archivo** **.csv** **(La Materia Prima):** **MUY IMPORTANTE.** **Necesito un archivo de resultados que haya botado** **MeasureTool**. Necesito ver los encabezados reales (¿se llaman **VelX**, **Vx**, **Velocity_X**?), en qué fila empiezan los datos, y ver exactamente el error de la "coma vs punto decimal". Con ese CSV, programo el script de Pandas (ETL) en 10 minutos.

---

### 2. Tu Entorno de Desarrollo (Lo que debes instalar en tu PC)

**Mientras consigues los archivos de Diego, esto es lo que necesitas tener instalado en tu laptop (tu ambiente de** **Staging**):

* **IDE (El Taller):** **Te recomiendo encarecidamente** **Cursor** **(basado en VS Code pero con IA nativa) o** **Visual Studio Code**.
* **Python 3.10 o superior:** **Instalado nativamente en Windows.**
* **El Motor DualSPHysics v5.2:** **Descargado y descomprimido en una carpeta (ej:** **C:\DualSPHysics**). Solo necesitamos los **.exe** **de la carpeta** **bin\windows** **para que nuestro código Python los mande a llamar.**
* **Librerías Clave (Las Herramientas):** **Abre tu terminal e instala este stack base:**

  **code**Bash

  ```
  pip install numpy pandas trimesh lxml scipy sqlalchemy
  ```

### ¿Cómo empezamos apenas tengas los archivos? (Plan de Acción Inmediato)

 **En cuanto tengas esos 4 archivos de Diego, me los describes o me pegas fragmentos aquí (especialmente el interior del** **.xml** **y el** **.bat**), y atacamos el **Sprint 1**:

* **Crearemos la estructura de carpetas de tu repositorio.**
* **Escribiremos el script** **geometry_builder.py** **para que lea el** **.stl** **y escupa las coordenadas exactas.**
* **Escribiremos el parser XML para inyectar esas coordenadas.**

**Conclusión:** **Pídele a Diego (o al profesor) ese ZIP con un caso de prueba funcional lo antes posible. En cuanto lo tengas en tu escritorio, arrancamos la línea de ensamblaje. ¿Dudas con el setup de tu PC o vamos por los archivos?**

Model

![Thinking](https://www.gstatic.com/aistudio/watermark/watermark.png)Thoughts

Expand to view model thoughts

chevron_right
