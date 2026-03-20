# scripts/build_prototypes.py
"""
Script para generar prototipos de conceptos a partir de vectores FastText.
Los prototipos se guardan en un archivo .pt para ser cargados posteriormente por el loader de neuronas.
"""

import torch
import sys
sys.path.append('.')
import logging
from pycore.utils.word_vectors import get_word_vectors, embed_text
# Configurar logger
logger = logging.getLogger(__name__)

# Cargar los vectores FastText (multilingüe o español según disponibilidad)
get_word_vectors("data/cc.multi.300.vec")  # o cc.es.300.vec

# Definir los conceptos y sus frases representativas
# IMPORTANTE: los IDs deben coincidir con los que usas en micro_ids y macro_ids
CONCEPTS = {
    # Conceptos sociales
    "greeting": ["hola", "buenos días", "saludos", "hello", "good morning"],
    "farewell": ["adiós", "hasta luego", "bye", "goodbye"],
    "wellbeing_question": ["cómo estás", "qué tal", "how are you"],
    "wellbeing_response": ["bien", "muy bien", "estoy bien", "I'm fine"],
    "gratitude": ["gracias", "thank you", "thanks"],
    "agreement": ["sí", "claro", "ok", "yes", "okay"],
    "disagreement": ["no", "nunca", "no way", "nope"],
    
    # Preguntas
    "identity_question": ["quién eres", "who are you"],
    "capability_question": ["qué puedes hacer", "what can you do"],
    "time_question": ["qué hora es", "what time is it"],
    "place_question": ["dónde estás", "where are you"],
    "reason_question": ["por qué", "why"],
    
    # Matemáticas
    "number": ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez"],
    "addition_operation": ["más", "plus", "suma"],
    "subtraction_operation": ["menos", "minus", "resta"],
    "multiplication_operation": ["por", "times", "multiplicación"],
    "division_operation": ["dividido", "divided by", "entre"],
    "calculation_question": ["cuánto es", "how much is"],
    "calculation_result": ["resultado", "result"],
    
    # Otros
    "clarification": ["aclara", "explícame", "clarify", "explain"],
    "conversation_start": ["empezar", "comencemos", "let's start"],
    "offer_help": ["ayuda", "help"],
    "weather": ["clima", "tiempo", "weather"],
    "time": ["tiempo", "hora", "time"],
    "external_identity": ["tú", "you"],
    "own_identity": ["yo", "I", "me"],
    "affirmation": ["sí", "claro", "yes"],
    "negation": ["no", "nunca", "never"],
}

# Diccionario para almacenar los prototipos generados
prototypes = {}

logger.debug("Generando prototipos para cada concepto...")

for concept_name, phrases in CONCEPTS.items():
    logger.debug(f"  Procesando concepto: {concept_name}")
    
    # Obtener embeddings para cada frase representativa
    phrase_vectors = []
    for phrase in phrases:
        vec = embed_text(phrase)
        phrase_vectors.append(vec)
    
    if phrase_vectors:
        # Calcular el prototipo como el promedio de todos los vectores de frases
        prototype = torch.stack(phrase_vectors).mean(dim=0)
        prototypes[concept_name] = prototype
    else:
        logger.debug(f"  ⚠️  Advertencia: No hay frases para el concepto {concept_name}, se omite")

# Guardar los prototipos en un archivo
output_file = "prototypes.pt"
torch.save(prototypes, output_file)
logger.debug(f"✅ Prototipos guardados en {output_file}")
logger.debug(f"   Total de conceptos procesados: {len(prototypes)}")

# Mostrar algunos ejemplos para verificar
logger.debug("\nEjemplos de prototipos generados:")
sample_concepts = list(prototypes.keys())[:5]
for concept in sample_concepts:
    vec = prototypes[concept]
    logger.debug(f"  {concept}: dimensión {vec.shape[0]}, norma {vec.norm().item():.4f}")