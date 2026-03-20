# 🧠 AURA – Arquitectura Cognitiva Jerárquica con Aprendizaje Continuo

AURA es un **sistema cognitivo artificial** inspirado en modelos conexionistas como **TRACE** y en los principios de la **energía libre** (*Free Energy Principle*) y la inferencia activa (Active Inference). Su objetivo es construir una inteligencia que aprenda de la interacción con el entorno, sin depender de modelos preentrenados externos, y que sea capaz de generalizar el conocimiento entre idiomas y dominios.

AURA es una **arquitectura cognitiva artificial** bioinspirada, diseñada para modelar procesos de percepción, abstracción, deliberación y expresión de manera unificada. A diferencia de los modelos convencionales de deep learning (como los transformers), AURA se basa en principios de **energía libre**, **código predictivo (predictive coding)**, **propagación de activación** y **aprendizaje Hebbiano local**, con el objetivo de construir sistemas que no solo procesen información, sino que **razonen, planifiquen y se adapten** continuamente.

A diferencia de los transformers y los LLMs tradicionales, AURA se basa en una arquitectura jerárquica de neuronas simbólicas y subsimbólicas con aprendizaje local, plasticidad estructural y mecanismos de homeostasis, todo ello implementado de forma eficiente con PyTorch y matrices dispersas.

## 🌟 Características principales

- **Arquitectura en tres niveles:**
  - **Micro‑neuronas:** representan palabras o conceptos básicos, con embeddings dinámicos que se actualizan con la experiencia.
  - **Neuronas:** detectan patrones y combinan micro‑neuronas mediante aprendizaje hebbiano y refuerzo.
  - **Macro‑neuronas:** conceptos abstractos (p.ej., saludo, pregunta_bienestar) que permiten la generalización y el multilingüismo.

- **Aprendizaje continuo:**
  - Hebb local para asociaciones.
  - Refuerzo interactivo a partir del feedback del usuario.
  - Homeostasis que ajusta umbrales según la frecuencia de activación.
  - Plasticidad estructural que crea nuevas conexiones cuando es necesario.

- **Multilingüismo emergente:**
  - Las palabras de distintos idiomas se asocian a los mismos conceptos universales (macro‑neuronas).
  - La gramática se aprende como una matriz de transiciones entre conceptos, no como reglas fijas.
  - Esto permite transferir conocimiento de un idioma a otro sin necesidad de reentrenamiento.

- **Generación de nuevos conceptos:**
  - Un módulo de clustering online detecta grupos de palabras que co‑ocurren frecuentemente y crea nuevos conceptos (macro‑neuronas) automáticamente.
  - El sistema puede expandir su vocabulario conceptual sin intervención humana.

- **Escalabilidad:**
  - Uso de tensores y matrices dispersas (COO, CSR) para manejar millones de neuronas.
  - Operaciones vectorizadas en PyTorch, con soporte para GPU.
  - Diseño preparado para computación distribuida y hardware neuromórfico.

## 🧩 Componentes principales

| Módulo | Descripción |
|--------|-------------|
| `micro_neuron.py` | Estado y funciones para las micro‑neuronas (embeddings, activación, decaimiento, homeostasis). |
| `neuron.py` | Estado y funciones para las neuronas intermedias (evaluación, inhibición lateral, aprendizaje hebbiano). |
| `macro_neuron.py` | Estado para las macro‑neuronas (conceptos), incluye la matriz de transiciones gramaticales. |
| `interconnector.py` | Conexiones entre conceptos (interconectoras) para enriquecer la representación. |
| `memory.py` | Memoria jerárquica (corto, medio, largo plazo) y recuperación asociativa mediante grafo conceptual. |
| `cognitive_engine.py` | Orquestador principal: gestiona el ciclo de razonamiento, feedback y plasticidad. |
| `thinking_neurons.py` | Thinking Neurons que generan planes conceptuales a partir del contexto (Social, Lógica, Ambigüedad). |
| `response_builder.py` | Genera respuestas en lenguaje natural mapeando conceptos a palabras según el idioma. |
| `context_synthesizer.py` | Infiere hipótesis de contexto a partir de las activaciones neuronales. |
| `concept_clustering.py` | Detecta y crea nuevos conceptos basándose en co‑ocurrencias. |
| `neuron_loader.py` | Carga y fusiona neuronas base, aprendidas y personalizadas desde diccionarios. |
| `base_neuron_data.py` | Datos iniciales del vocabulario (español, inglés, matemáticas, etc.) con metadatos de idioma y concepto. |
| `learned_neuron_data.py` | Datos de neuronas aprendidas durante la interacción (patrones, etc.). |
| `personality_neurons.py` | Neuronas personalizadas que definen la identidad y rasgos de Aura. |
| `word_vectors.py` | Cargador de vectores FastText para embeddings semánticos. |
| `vector_index.py` | Índice vectorial para búsqueda por similitud (basado en PyTorch). |
| `config.py` | Gestor de configuración (carga desde YAML). |

## 🏗️ Arquitectura general

```text
Entrada (texto)
↓
[Capa 1: Percepción]
  Embedding semántico (FastText)
  Activación de micro‑neuronas (similitud coseno)
  Evaluación de neuronas (patrones) con inhibición lateral y feedback
↓
[Capa 2: Abstracción]
  Activación de macro‑neuronas (conceptos) basada en condiciones de neuronas
  Matriz de transiciones entre conceptos (para gramática aprendida)
↓
[Capa 3: Deliberación]
  Sintetizador de Contexto → hipótesis
  Thinking Neurons → propuestas de planes conceptuales
  MacroTN → selección del plan ganador
↓
[Capa 4: Expresión]
  ResponseBuilder → genera respuesta (plantilla, cálculo o concatenación)
↓
Salida (texto)

## 🚀 Estado del proyecto

✅ Motor de activación y razonamiento funcional.
✅ Aprendizaje hebbiano, por refuerzo y homeostasis implementados.
✅ Soporte multilingüe básico (español/inglés) con conceptos universales.
✅ Matriz de transiciones gramaticales aprendida.
✅ Mecanismo de creación de nuevos conceptos (clustering online) en desarrollo.
⚠️ Pendiente: integración completa con el Brain v1 (dinámica continua, energía libre, meta‑control).
⚠️ Pendiente: escalado distribuido y soporte para hardware neuromórfico.

📦 Instalación

## 📦 Instalación

```bash
git clone https://github.com/tuusuario/aura-cognitive-architecture.git
cd aura-cognitive-architecture
pip install -r requirements.txt
```

## Requisitos
```bash
Python 3.10+
PyTorch 2.0+
numpy
pyyaml
gensim
```
## 🚀 Uso básico

Para iniciar una conversación con AURA en español:

```bash
python main.py --lang es
Opciones disponibles:

--device cpu|cuda – selecciona dispositivo.

--config ruta.yaml – archivo de personalidad (YAML).

--lang es|en – idioma de la respuesta.

--no-response – solo muestra activaciones, sin generar respuesta.
```

Durante la conversación, puedes escribir:

- Saludos (hola, buenos días) → respuesta social.
- Preguntas simples (¿cómo estás?, ¿quién eres?) → respuestas predefinidas.
- Operaciones matemáticas (2+3, 5-1) → evaluación y resultado.
- Frases ambiguas o desconocidas → el sistema puede responder "aclara" o "no estoy seguro".

El feedback del usuario (s/n) se utiliza para reforzar o debilitar las conexiones neuronales, permitiendo un aprendizaje por refuerzo básico.

## 📚 Ejemplo de conversación

```text

Tú: hola
Aura: ¡Hola! ¿Cómo estás?

Tú: bien, ¿y tú?
Aura: Estoy bien, gracias por preguntar.

Tú: what is 2+2?
Aura: The result is 4.
```

## 🧠 Filosofía del proyecto

AURA no es un LLM. No pretende competir con ChatGPT en términos de escala bruta, sino explorar una vía diferente: inteligencia que emerge de la interacción, la plasticidad local y la abstracción conceptual. Creemos que este enfoque puede conducir a sistemas más interpretables, adaptables y eficientes energéticamente, y eventualmente a una verdadera inteligencia artificial general (AGI).

## 🤝 Contribuciones

El proyecto está en fase activa de desarrollo. Las contribuciones son bienvenidas, especialmente en áreas como:

- Nuevas Thinking Neurons para dominios específicos.
- Mejoras en la eficiencia de las operaciones dispersas.
- Integración con otros modelos de embeddings o modalidades (imagen, audio).
- Implementación de mecanismos de plasticidad estructural más avanzados.

Por favor, abre un issue para discutir cambios importantes antes de enviar un pull request.

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo LICENSE para más detalles.

## 📬 Contacto

Autor: J.J. Guti
Email: jogugil@gmail.com
GitHub: @jogugil

Si tienes preguntas o quieres colaborar, puedes escribir a Faasioflex@gmail.com o abrir un issue en GitHub.

## 🌟 Aghradecimientos

Este trabajo se inspira en investigaciones de Karl Friston (energía libre), Rao & Ballard (predictive coding), y en arquitecturas cognitivas como ACT‑R, SOAR y LIDA. También agradecemos a la comunidad de código abierto por las herramientas que hacen posible este proyecto.

¡Gracias por tu interés en AURA! 🌟


