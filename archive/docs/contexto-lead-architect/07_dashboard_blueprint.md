# SMART DASHBOARD & VISUALIZATION BLUEPRINT

## 1. VISION ESTRATEGICA DEL DASHBOARD

El Dashboard no es solo un visualizador de datos; es la **interfaz del Modelo Surrogate**. Su objetivo es permitir que un usuario (Ingeniero o Investigador) interactue con el algoritmo de Machine Learning para obtener predicciones instantaneas sobre la estabilidad de los bloques costeros, eliminando la necesidad de ejecutar DualSPHysics para cada nuevo escenario.

* **Stack Tecnologico:** Python, Streamlit, Plotly, SQLAlchemy.
* **Fuente de Datos:** `results.sqlite` (Datos historicos) y `GPR_model.pkl` (Pesos del modelo entrenado).
* **URL de Produccion (Conceptual):** `https://incipient-motion-pred.streamlit.app`

---

## 2. ARQUITECTURA DE LA INTERFAZ (UI/UX)

### A. Sidebar: Panel de Control Parametrico (Inputs)

El usuario define las condiciones de contorno del tsunami y las propiedades del bloque.

1. **Selector de Geometria:** Menu desplegable con los 7 tipos de bloques de Bahia Cisnes (BLIR1 a BLIR7). Incluye una miniatura 3D del STL seleccionado.
2. **Slider de Masa (M):** Rango definido por el Dr. Moris (ej. 0.5 kg a 50 kg).
3. **Slider de Altura de Ola (H):** Rango de inundacion (ej. 0.1 m a 1.0 m).
4. **Slider de Angulo de Incidencia (theta):** Rotacion inicial del bloque respecto al flujo (0 a 360 grados).
5. **Selector de dp (Resolucion de Referencia):** Para elegir contra que nivel de fidelidad se quiere comparar la prediccion (ej. 0.005m).

### B. Main Panel: Prediccion de Inteligencia Artificial (Modulo 4)

Muestra el resultado procesado por el **Gaussian Process Regressor (GPR)** en tiempo real.

1. **Indicador de Estado (Semaforo de Estabilidad):**
   * **VERDE (Safe):** El bloque permanecera estable bajo estas condiciones.
   * **ROJO (Critical):** Se predice movimiento incipiente o transporte masivo.
2. **Gauge de Probabilidad:** Un grafico circular que muestra la probabilidad de fallo (0% a 100%).
3. **Visualizacion de Incertidumbre (sigma):** Un indicador que dice que tan "segura" esta la IA de su respuesta.
   * *Logica:* Si el punto consultado por el usuario esta lejos de los datos simulados en la RTX 5090, la incertidumbre sube, sugiriendo la necesidad de una nueva simulacion real.

### C. Panel de Ingenieria: Analisis FSI (Fisica Real)

Visualiza los datos extraidos por el Modulo 3 (`data_cleaner.py`) de las simulaciones que ya existen en la base de datos.

1. **Series de Tiempo (Plotly):**
   * Grafico 1: Fuerza de Impacto SPH vs Fuerza de Contacto Chrono (N).
   * Grafico 2: Velocidad del bloque (m/s) vs Tiempo (s).
2. **Trayectoria 3D:** Un grafico de dispersion 3D que muestra el recorrido del centro de masa del bloque sobre la rampa de acero.
3. **Diagrama de Fase (Frontera de Estabilidad):** Un grafico 2D (M vs H) donde una linea curva (la frontera) separa los casos estables de los inestables. El punto actual del usuario se marca sobre este mapa.

---

## 3. LOGICA DE BACK-END (INTEGRACION)

### A. Conectividad con SQLite

El dashboard debe realizar consultas SQL dinamicas para filtrar los tsunamis similares al que el usuario esta configurando.

* *Query de ejemplo:* `SELECT * FROM convergence WHERE boulder_mass BETWEEN x AND y AND dam_height BETWEEN a AND b`.

### B. Ejecucion del Modelo Surrogate

En lugar de llamar a DualSPHysics, el dashboard carga el modelo entrenado de Scikit-Learn:

```python
prediction, sigma = gpr_model.predict(user_inputs, return_std=True)
```

Esto permite una latencia de respuesta de **< 100 milisegundos**, transformando un proceso de 24 horas de GPU en una herramienta de consulta instantanea.

---

## 4. FUNCIONALIDADES DE EXPORTACION (PREPARACION PARA EL PAPER Q1)

El dashboard debe incluir botones de exportacion para generar los activos del articulo cientifico:

1. **Export Figure:** Descarga de los graficos en formato `.pdf` o `.svg` con resolucion de 300 DPI.
2. **Export CSV:** Descarga de la tabla filtrada de resultados para analisis estadistico externo.
3. **Generate Report:** Un boton que genera un resumen tecnico en Markdown con los resultados de la prediccion y los parametros fisicos asociados.

---

## 5. REQUERIMIENTOS DE DESPLIEGUE (DEPLOYMENT)

* **Entorno Local:** `streamlit run src/dashboard/app.py`.
* **Optimizacion de Memoria:** Uso de `@st.cache_resource` para cargar la base de datos SQLite y el modelo ML una sola vez al inicio, evitando latencia en la interaccion del usuario.
* **Seguridad:** Implementacion de un Login simple si el Dr. Moris desea mantener los datos del Fondecyt privados antes de la publicacion del paper.
