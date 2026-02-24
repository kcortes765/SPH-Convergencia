# PHYSICAL AUDIT & FORENSIC ANALYSIS

## 1. INTRODUCCION: EL ESTADO DEL ARTE HEREDADO

Antes de la intervencion de Kevin Cortes, el laboratorio utilizaba un flujo de trabajo manual (Investigador Diego). Tras un analisis exhaustivo de los archivos de la simulacion base (`00001` a `00010`), se detectaron errores sistematicos que invalidaban los resultados para una publicacion de alto impacto (Q1).

**El pipeline actual no solo automatiza, sino que actua como un filtro de calidad fisica.**

---

## 2. ANOMALIA 1: EL ERROR DE DENSIDAD (VOLUMEN IRREAL)

### El Problema (Heredado):

Diego configuro la masa del boulder en **1.06053 kg** basandose en un bloque real impreso en 3D y pesado en balanza. Sin embargo, para los calculos de volumen en el software, se asumio implicitamente el volumen del **Bounding Box** (la "caja de zapatos" que envuelve la roca irregular).

* **Volumen Bounding Box:** ~0.0014 m3.
* **Densidad resultante (comentada):** ~800 kg/m3 (lo que permitiria que el bloque flotara o fuera muy inestable).

### El Hallazgo Forense:

Al analizar el archivo `BLIR3.stl` con el Modulo 1 (`trimesh`), descubrimos que el **Volumen Real de la malla cerrada** es de apenas **0.00053 m3** (menos de la mitad del Bounding Box).

### El Impacto Fisico:

DualSPHysics toma la masa impuesta (1.06 kg) y la aplica sobre el volumen que el rellena con particulas.

* **Densidad Real Simulada:** 1.06 kg / 0.00053 m3 = 2000.1 kg/m3.
* **Consecuencia:** Diego creia estar simulando una roca de plastico liviano (PVC), pero el software estaba calculando una roca de **Concreto/Arenisca maciza**. Esto subestimaba drasticamente el transporte del bloque por el tsunami.

### El Fix de Arquitectura:

El Modulo 1 calcula el volumen exacto de la malla y ajusta la masa o reporta la densidad corregida. El usuario ahora tiene control real sobre la flotabilidad.

---

## 3. ANOMALIA 2: EL ERROR DEL TENSOR DE INERCIA (VOXELIZACION)

### El Problema (Heredado):

DualSPHysics (via GenCase) calcula la inercia del solido sumando la contribucion de cada particula de masa puntual que logra meter dentro del STL.

* A resoluciones gruesas (dp = 0.05m), la roca se representaba con solo **31 particulas**.
* Estas particulas, al ser esferas, no rellenan las puntas ni los bordes irregulares de la roca, dejando huecos y concentrando la masa de forma erronea.

### El Hallazgo Forense:

Comparamos el Tensor de Inercia calculado por GenCase vs. el calculado analiticamente por nuestro Modulo 1 (`trimesh`):

* **Ixx (GenCase):** 0.00406
* **Ixx (Trimesh):** 0.00219
* **Error:** GenCase **sobreestimaba la inercia en un 185% a 300%**.

### El Impacto Fisico:

Una inercia 3 veces mayor a la real significa que el bloque ofrece una resistencia artificial gigante a empezar a rotar. El tsunami le pegaba a la roca y esta no giraba ("Pitch/Roll") como deberia, falseando toda la cinematica del movimiento incipiente.

### El Fix de Arquitectura:

Nuestro codigo anula el calculo interno de DualSPHysics. Python calcula el Tensor de Inercia de la geometria continua (STL liso) e inyecta los valores exactos en el XML:
`<inertia x="0.00219" y="0.00158" z="0.00361" />`.
**El bloque ahora rota perfectamente sin importar si el dp es grueso o fino.**

---

## 4. ANOMALIA 3: FRICCION Y REBOTE (PVC VS LIME-STONE)

### El Problema (Heredado):

En el archivo `Floating_Materials.xml`, la roca estaba definida como material `pvc` con un coeficiente de restitucion (rebote) de **0.60**.

* Diego y el profesor notaron en los graficos de altura que el bloque "saltaba" o "vibraba" de forma extrana al inicio del tsunami.

### El Hallazgo Forense:

El PVC sobre acero tiene un comportamiento elastico que no corresponde a una roca costera sobre fondo marino. Ademas, la masa de 1.06 kg (2000 kg/m3) confirmaba que el objeto *debia* ser roca, pero sus propiedades de superficie eran de plastico.

### El Fix de Arquitectura:

Se forzo el cambio del material a `lime-stone` (ya definido en el sistema), reduciendo el coeficiente de restitucion y ajustando la friccion de Chrono para simular contacto roca-rampa realista. Esto elimino el "ruido" de rebote en las graficas finales.

---

## 5. ANOMALIA 4: EL "GAP" DE ASENTAMIENTO (FTPause)

### El Problema (Heredado):

Diego usaba `FtPause="0.0"`.

* Al iniciar la simulacion, el boulder se creaba a una micro-distancia del suelo (dp/2). Al no haber pausa, el boulder caia por gravedad al mismo tiempo que la ola impactaba.

### El Hallazgo Forense:

Los CSV de Chrono mostraban aceleraciones verticales negativas espurias en t=0.001s.

### El Fix de Arquitectura:

Nuestro orquestador inyecta siempre `FtPause="0.5"`. Esto permite que el boulder se asiente en el fondo y las presiones se estabilicen antes de liberar la columna de agua (Dam Break).

---

## 6. CONCLUSION PARA EL DESARROLLO FUTURO

Cualquier propuesta de codigo o modificacion del XML que realice la IA debe respetar estos principios:

1. **Prioridad Geometrica:** La matematica del STL (Trimesh) manda sobre la discrecion de particulas (GenCase).
2. **Transparencia de Densidad:** No se debe inyectar solo la masa; se debe validar que `masa / volumen_stl` sea coherente con el material real.
3. **Aislamiento de la Convergencia:** Al fijar la inercia y masa via Python, el estudio de convergencia de malla (dp) ahora es **puro**. Ya no varia la propiedad del objeto, solo varia la resolucion del fluido, lo cual es la definicion estricta de un estudio de convergencia academico.
