# learned_neuron_data.py
# Devuelve una lista de micro‑neuronas aprendidas (tipo 'pattern').

from typing import List, Dict, Any

def get_learned_neurons () -> List[Dict[str, Any]]:
    """
    Devuelve una lista de diccionarios, cada uno con la información para
    crear una micro‑neurona aprendida (tipo 'pattern').
    """
    learned = []

    # Ejemplo 1: patrón "qué tal"
    learned.append({
        "id": "n_patron_que_tal",
        "concept": "qué tal",
        "type": "pattern",
        "metadata": {
            "sequence": ["que", "tal"],
            "regex": r"que tal",
            "activation": {
                "starts_phrase": True,
                "threshold": 0.95,
                "contexts": ["conversation", "greeting"]
            },
            "meanings": {
                "default": ["casual_greeting"],
                "with_heat": ["weather_comment"]
            },
            "macro_tags": ["macro_greeting", "macro_pattern"],
            "semantic_fields": ["greeting", "social", "pattern"],
            "is_greeting": 1,
            "is_question": 0,
            "is_macro_of": ["macro_greeting"],
            "synonyms": ["mn_que_hay"],
            "variants": ["mn_que_tal_amigo"],
            "personality": {
                "tone": "friendly",
                "emotion": "positive",
                "register": "informal"
            },
            "explanation": "Secuencia usada para saludar de manera casual.",
            "examples": ["¡Qué tal!", "¡Hola, qué tal?"],
            "origin": "user_taught"
        }
    })

    # Ejemplo 2: patrón "hace calor"
    learned.append({
        "id": "n_patron_hace_calor",
        "concept": "hace calor",
        "type": "pattern",
        "metadata": {
            "sequence": ["hace", "calor"],
            "regex": r"hace calor",
            "activation": {
                "threshold": 0.9,
                "contexts": ["weather", "conversation"]
            },
            "meanings": {
                "default": ["weather_comment"],
                "with_question": ["status_question"]
            },
            "macro_tags": ["macro_weather", "macro_pattern"],
            "semantic_fields": ["weather", "comment", "pattern"],
            "is_greeting": 0,
            "is_question": 0,
            "is_macro_of": ["macro_weather"],
            "synonyms": ["mn_que_calor"],
            "variants": [],
            "personality": {
                "tone": "neutral",
                "emotion": "neutral",
                "register": "informal"
            },
            "explanation": "Secuencia usada para comentar sobre el clima.",
            "examples": ["¡Hace calor!", "Hoy hace calor."],
            "origin": "self_learned"
        }
    })

    return learned 