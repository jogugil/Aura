# base_neuron_data.py
# Versión adaptada: devuelve diccionarios con los datos para poblar micro‑neuronas y neuronas.
# Incluye información de idioma y asociación a conceptos (macro‑neuronas) para multilingüismo.
# Ampliado con vocabulario en inglés, conceptos matemáticos y palabras de clarificación.

from typing import List, Dict, Any

def get_base_data() -> Dict[str, List]:
    """
    Devuelve un diccionario con las listas de datos para construir
    el estado inicial de micro‑neuronas y neuronas, y también macro‑neuronas.
    """
    # Mapeo de campo semántico a ID de concepto (macro‑neurona)
    field_to_concept = {
        # Social
        'saludo': 'concepto_saludo',
        'despedida': 'concepto_despedida',
        'pregunta_bienestar': 'concepto_pregunta_bienestar',
        'pregunta_reciproca': 'concepto_pregunta_reciproca',
        'agradecimiento': 'concepto_agradecimiento',
        'respuesta_agradecimiento': 'concepto_respuesta_agradecimiento',
        'afirmacion': 'concepto_acuerdo',
        'negacion': 'concepto_desacuerdo',
        'cortesia': None,
        # Preguntas
        'pregunta_general': 'concepto_peticion_info_general',
        'pregunta_identidad': 'concepto_pregunta_identidad',
        'pregunta_tiempo': 'concepto_pregunta_tiempo',
        'pregunta_lugar': 'concepto_pregunta_lugar',
        'pregunta_razon': 'concepto_pregunta_razon',
        'pregunta_especifica': 'concepto_peticion_info_especifica',
        'pregunta_calculo': 'concepto_pregunta_calculo',
        # Identidad
        'identidad_propia': 'concepto_auto_revelacion_identidad',
        'identidad_externa': 'concepto_identidad_externa',
        'auto_revelacion_capacidad': 'concepto_auto_revelacion_capacidad',
        # Respuestas
        'respuesta_bienestar': 'concepto_respuesta_bienestar',
        'respuesta_calculo': 'concepto_resultado_calculo',
        # Matemáticas
        'numero': 'concepto_numero',
        'operacion_suma': 'concepto_operacion_suma',
        'operacion_resta': 'concepto_operacion_resta',
        'operacion_multiplicacion': 'concepto_operacion_multiplicacion',
        'operacion_division': 'concepto_operacion_division',
        # Clarificación (nuevo)
        'clarificacion': 'concepto_clarificacion',
        # Otros
        'asistencia': 'concepto_ofrecer_ayuda',
        'ambiente': 'concepto_clima',
        'temporalidad': 'concepto_tiempo',
        'intensificador': None,
        'rasgo_positivo': None,
        'conector_discursivo': None,
    }

    # ---- Micro‑neuronas: vocabulario unificado ----
    unified_vocabulary = [
        # ========== ESPAÑOL ==========
        # --- Saludos y despedidas ---
        ("mn_hola", "hola", {'idioma': 'es', 'semantic_field': 'saludo', 'personalidad': {'tono': 'casual', 'emocion': 'neutral'}, 'GRAMATICA': {'TIPO': 'saludo', 'PERMITE_DESPUES': ['interrogativo_estado', 'fin_frase']}}),
        ("mn_buenos", "buenos", {'idioma': 'es', 'GRAMATICA': {'TIPO': 'parte_saludo_formal'}}),
        ("mn_dias", "días", {'idioma': 'es', 'GRAMATICA': {'TIPO': 'parte_saludo_formal'}}),
        ("mn_buenas", "buenas", {'idioma': 'es', 'GRAMATICA': {'TIPO': 'parte_saludo_formal'}}),
        ("mn_tardes", "tardes", {'idioma': 'es', 'GRAMATICA': {'TIPO': 'parte_saludo_formal'}}),
        ("mn_noches", "noches", {'idioma': 'es', 'GRAMATICA': {'TIPO': 'parte_saludo_formal'}}),
        ("mn_adios", "adiós", {'idioma': 'es', 'semantic_field': 'despedida', 'personalidad': {'tono': 'neutral', 'emocion': 'neutral'}, 'GRAMATICA': {'TIPO': 'despedida', 'PERMITE_DESPUES': ['fin_frase']}}),
        ("mn_chao", "chao", {'idioma': 'es', 'semantic_field': 'despedida', 'GRAMATICA': {'TIPO': 'despedida_informal'}}),
        ("mn_hasta_luego", "hasta luego", {'idioma': 'es', 'semantic_field': 'despedida', 'GRAMATICA': {'TIPO': 'despedida_compuesta'}}),
        ("mn_nos_vemos", "nos vemos", {'idioma': 'es', 'semantic_field': 'despedida', 'GRAMATICA': {'TIPO': 'despedida_compuesta'}}),

        # --- Interrogativos ---
        ("mn_que", "qué", {'idioma': 'es', 'semantic_field': 'pregunta_general', 'personalidad': {'tono': 'neutral'}, 'GRAMATICA': {'TIPO': 'interrogativo_general', 'PERMITE_DESPUES': ['sustantivo', 'verbo']}}),
        ("mn_q", "q", {'idioma': 'es', 'semantic_field': 'pregunta_general', 'GRAMATICA': {'TIPO': 'interrogativo_informal'}}),
        ("mn_quien", "quién", {'idioma': 'es', 'semantic_field': 'pregunta_identidad', 'GRAMATICA': {'TIPO': 'interrogativo_persona'}}),
        ("mn_como", "cómo", {'idioma': 'es', 'semantic_field': 'pregunta_bienestar', 'personalidad': {'tono': 'neutral'}, 'GRAMATICA': {'TIPO': 'interrogativo_estado', 'PERMITE_DESPUES': ['verbo_estado']}}),
        ("mn_cuando", "cuándo", {'idioma': 'es', 'semantic_field': 'pregunta_tiempo', 'GRAMATICA': {'TIPO': 'interrogativo_temporal'}}),
        ("mn_donde", "dónde", {'idioma': 'es', 'semantic_field': 'pregunta_lugar', 'GRAMATICA': {'TIPO': 'interrogativo_lugar'}}),
        ("mn_por_que", "por qué", {'idioma': 'es', 'semantic_field': 'pregunta_razon', 'GRAMATICA': {'TIPO': 'interrogativo_causal'}}),
        ("mn_cual", "cuál", {'idioma': 'es', 'semantic_field': 'pregunta_especifica', 'GRAMATICA': {'TIPO': 'interrogativo_seleccion'}}),
        ("mn_tal", "tal", {'idioma': 'es', 'semantic_field': 'pregunta_bienestar', 'GRAMATICA': {'TIPO': 'adverbio_interrogativo'}}),

        # --- Verbos ---
        ("mn_ser", "ser", {'idioma': 'es', 'GRAMATICA': {'TIPO': 'verbo_infinitivo', 'RAIZ': 'ser'}}),
        ("mn_eres", "eres", {'idioma': 'es', 'GRAMATICA': {'TIPO': 'verbo_estado', 'PERSONA': 2, 'TIEMPO': 'presente'}}),
        ("mn_es", "es", {'idioma': 'es', 'GRAMATICA': {'TIPO': 'verbo_estado', 'PERSONA': 3, 'TIEMPO': 'presente'}}),
        ("mn_soy", "soy", {'idioma': 'es', 'semantic_field': 'identidad_propia', 'personalidad': {'tono': 'neutral'}, 'GRAMATICA': {'TIPO': 'verbo_estado', 'REQUIERE': ['pronombre'], 'PERMITE_DESPUES': ['sustantivo', 'adjetivo'], 'PROHIBE_DESPUES': ['verbo_estado']}}),
        ("mn_somos", "somos", {'idioma': 'es', 'GRAMATICA': {'TIPO': 'verbo_estado', 'PERSONA': 1, 'PLURAL': True, 'TIEMPO': 'presente'}}),
        ("mn_estar", "estar", {'idioma': 'es', 'GRAMATICA': {'TIPO': 'verbo_infinitivo', 'RAIZ': 'estar'}}),
        ("mn_estas", "estás", {'idioma': 'es', 'semantic_field': 'pregunta_bienestar', 'personalidad': {'tono': 'neutral'}, 'GRAMATICA': {'TIPO': 'verbo_estado', 'REQUIERE': ['interrogativo_estado'], 'PERMITE_DESPUES': ['fin_frase']}}),
        ("mn_esta", "está", {'idioma': 'es', 'GRAMATICA': {'TIPO': 'verbo_estado', 'PERSONA': 3, 'TIEMPO': 'presente'}}),
        ("mn_estoy", "estoy", {'idioma': 'es', 'semantic_field': 'respuesta_bienestar', 'personalidad': {'tono': 'neutral'}, 'GRAMATICA': {'TIPO': 'verbo_estado', 'REQUIERE': ['pronombre'], 'PERMITE_DESPUES': ['adverbio_estado', 'adjetivo'], 'PROHIBE_DESPUES': ['verbo_estado']}}),
        ("mn_tener", "tener", {'idioma': 'es', 'GRAMATICA': {'TIPO': 'verbo_infinitivo', 'RAIZ': 'tener'}}),
        ("mn_tienes", "tienes", {'idioma': 'es', 'GRAMATICA': {'TIPO': 'verbo_posesion', 'PERSONA': 2, 'TIEMPO': 'presente'}}),
        ("mn_tengo", "tengo", {'idioma': 'es', 'GRAMATICA': {'TIPO': 'verbo_posesion', 'PERSONA': 1, 'TIEMPO': 'presente'}}),
        ("mn_hacer", "hacer", {'idioma': 'es', 'GRAMATICA': {'TIPO': 'verbo_infinitivo', 'RAIZ': 'hacer'}}),
        ("mn_haces", "haces", {'idioma': 'es', 'GRAMATICA': {'TIPO': 'verbo_accion', 'PERSONA': 2, 'TIEMPO': 'presente'}}),
        ("mn_hago", "hago", {'idioma': 'es', 'GRAMATICA': {'TIPO': 'verbo_accion', 'PERSONA': 1, 'TIEMPO': 'presente'}}),
        ("mn_poder", "poder", {'idioma': 'es', 'GRAMATICA': {'TIPO': 'verbo_infinitivo', 'RAIZ': 'poder'}}),
        ("mn_puedes", "puedes", {'idioma': 'es', 'GRAMATICA': {'TIPO': 'verbo_modal', 'PERSONA': 2, 'TIEMPO': 'presente'}}),
        ("mn_puedo", "puedo", {'idioma': 'es', 'semantic_field': 'auto_revelacion_capacidad', 'personalidad': {'tono': 'neutral'}, 'GRAMATICA': {'TIPO': 'verbo_modal', 'PERMITE_DESPUES': ['verbo_infinitivo']}}),
        ("mn_querer", "querer", {'idioma': 'es', 'GRAMATICA': {'TIPO': 'verbo_infinitivo', 'RAIZ': 'querer'}}),
        ("mn_quieres", "quieres", {'idioma': 'es', 'GRAMATICA': {'TIPO': 'verbo_deseo', 'PERSONA': 2, 'TIEMPO': 'presente'}}),
        ("mn_quiero", "quiero", {'idioma': 'es', 'GRAMATICA': {'TIPO': 'verbo_deseo', 'PERSONA': 1, 'TIEMPO': 'presente'}}),

        # --- Afirmación / Negación / Cortesía ---
        ("mn_si", "sí", {'idioma': 'es', 'semantic_field': 'afirmacion', 'GRAMATICA': {'TIPO': 'afirmacion'}}),
        ("mn_no", "no", {'idioma': 'es', 'semantic_field': 'negacion', 'GRAMATICA': {'TIPO': 'negacion'}}),
        ("mn_claro", "claro", {'idioma': 'es', 'semantic_field': 'afirmacion', 'personalidad': {'tono': 'positivo', 'emocion': 'seguridad'}, 'GRAMATICA': {'TIPO': 'afirmacion', 'PERMITE_DESPUES': ['pronombre', 'fin_frase']}}),
        ("mn_ok", "ok", {'idioma': 'es', 'semantic_field': 'afirmacion', 'GRAMATICA': {'TIPO': 'afirmacion_informal'}}),
        ("mn_gracias", "gracias", {'idioma': 'es', 'semantic_field': 'agradecimiento', 'personalidad': {'tono': 'positivo', 'emocion': 'positiva'}, 'GRAMATICA': {'TIPO': 'cortesia', 'PERMITE_DESPUES': ['fin_frase']}}),

        # --- Generales ---
        ("mn_nombre", "nombre", {'idioma': 'es', 'semantic_field': 'identidad', 'GRAMATICA': {'TIPO': 'sustantivo'}}),
        ("mn_tiempo", "tiempo", {'idioma': 'es', 'semantic_field': 'temporalidad', 'GRAMATICA': {'TIPO': 'sustantivo'}}),
        ("mn_clima", "clima", {'idioma': 'es', 'semantic_field': 'ambiente', 'GRAMATICA': {'TIPO': 'sustantivo'}}),
        ("mn_ayuda", "ayuda", {'idioma': 'es', 'semantic_field': 'asistencia', 'GRAMATICA': {'TIPO': 'sustantivo_o_verbo'}}),
        ("mn_tu", "tú", {'idioma': 'es', 'semantic_field': 'identidad_externa', 'GRAMATICA': {'TIPO': 'pronombre_personal'}}),
        ("mn_yo", "yo", {'idioma': 'es', 'semantic_field': 'identidad_propia', 'personalidad': {'tono': 'neutral'}, 'GRAMATICA': {'TIPO': 'pronombre', 'PERMITE_DESPUES': ['verbo_estado', 'verbo_accion']}}),

        # --- Matemáticas (español) ---
        ("mn_cero", "cero", {'idioma': 'es', 'semantic_field': 'numero', 'valor': 0}),
        ("mn_uno", "uno", {'idioma': 'es', 'semantic_field': 'numero', 'valor': 1}),
        ("mn_dos", "dos", {'idioma': 'es', 'semantic_field': 'numero', 'valor': 2}),
        ("mn_tres", "tres", {'idioma': 'es', 'semantic_field': 'numero', 'valor': 3}),
        ("mn_cuatro", "cuatro", {'idioma': 'es', 'semantic_field': 'numero', 'valor': 4}),
        ("mn_cinco", "cinco", {'idioma': 'es', 'semantic_field': 'numero', 'valor': 5}),
        ("mn_seis", "seis", {'idioma': 'es', 'semantic_field': 'numero', 'valor': 6}),
        ("mn_siete", "siete", {'idioma': 'es', 'semantic_field': 'numero', 'valor': 7}),
        ("mn_ocho", "ocho", {'idioma': 'es', 'semantic_field': 'numero', 'valor': 8}),
        ("mn_nueve", "nueve", {'idioma': 'es', 'semantic_field': 'numero', 'valor': 9}),
        ("mn_diez", "diez", {'idioma': 'es', 'semantic_field': 'numero', 'valor': 10}),
        ("mn_mas", "más", {'idioma': 'es', 'semantic_field': 'operacion_suma', 'simbolo': '+', 'GRAMATICA': {'TIPO': 'operador'}}),
        ("mn_menos", "menos", {'idioma': 'es', 'semantic_field': 'operacion_resta', 'simbolo': '-', 'GRAMATICA': {'TIPO': 'operador'}}),
        ("mn_por", "por", {'idioma': 'es', 'semantic_field': 'operacion_multiplicacion', 'simbolo': '*', 'GRAMATICA': {'TIPO': 'operador'}}),
        ("mn_dividido", "dividido", {'idioma': 'es', 'semantic_field': 'operacion_division', 'simbolo': '/', 'GRAMATICA': {'TIPO': 'operador'}}),
        ("mn_cuanto_es", "cuánto es", {'idioma': 'es', 'semantic_field': 'pregunta_calculo', 'GRAMATICA': {'TIPO': 'pregunta_calculo'}}),
        ("mn_resultado", "resultado", {'idioma': 'es', 'semantic_field': 'respuesta_calculo', 'GRAMATICA': {'TIPO': 'sustantivo'}}),

        # --- Palabras de clarificación (español) - NUEVAS ---
        ("mn_aclarar", "aclara", {'idioma': 'es', 'semantic_field': 'clarificacion', 'personalidad': {'tono': 'neutral'}, 'GRAMATICA': {'TIPO': 'verbo_imperativo'}}),
        ("mn_explicame", "explícame", {'idioma': 'es', 'semantic_field': 'clarificacion', 'personalidad': {'tono': 'neutro'}, 'GRAMATICA': {'TIPO': 'verbo_imperativo'}}),

        # --- Generativas (español) ---
        ("gen_y_tu", "y tú?", {'idioma': 'es', 'semantic_field': 'pregunta_reciproca', 'personalidad': {'tono': 'casual', 'emocion': 'curiosidad'}, 'GRAMATICA': {'TIPO': 'pregunta_reciproca', 'REQUIERE': ['adverbio_estado'], 'PERMITE_DESPUES': ['fin_frase']}}),
        ("gen_ayudarte", "ayudarte", {'idioma': 'es', 'semantic_field': 'asistencia', 'personalidad': {'tono': 'proactivo', 'emocion': 'positiva'}, 'GRAMATICA': {'TIPO': 'verbo_infinitivo_complejo', 'REQUIERE': ['verbo_modal'], 'PERMITE_DESPUES': ['fin_frase']}}),
        ("gen_analizar", "analizar", {'idioma': 'es', 'semantic_field': 'auto_revelacion_capacidad', 'GRAMATICA': {'TIPO': 'verbo_infinitivo'}}),
        ("gen_aprender", "aprender", {'idioma': 'es', 'semantic_field': 'auto_revelacion_capacidad', 'GRAMATICA': {'TIPO': 'verbo_infinitivo'}}),
        ("gen_procesar", "procesar", {'idioma': 'es', 'semantic_field': 'auto_revelacion_capacidad', 'GRAMATICA': {'TIPO': 'verbo_infinitivo'}}),
        ("gen_ia", "una IA", {'idioma': 'es', 'semantic_field': 'identidad_propia', 'personalidad': {'tono': 'neutral'}, 'GRAMATICA': {'TIPO': 'sustantivo', 'REQUIERE': ['verbo_estado'], 'PERMITE_DESPUES': ['adjetivo_calificativo', 'fin_frase']}}),
        ("gen_texto", "texto", {'idioma': 'es', 'semantic_field': 'auto_revelacion_capacidad', 'GRAMATICA': {'TIPO': 'sustantivo'}}),
        ("gen_lenguaje", "lenguaje", {'idioma': 'es', 'semantic_field': 'auto_revelacion_capacidad', 'GRAMATICA': {'TIPO': 'sustantivo'}}),
        ("gen_bien", "bien", {'idioma': 'es', 'semantic_field': 'respuesta_bienestar', 'personalidad': {'tono': 'positivo', 'emocion': 'positiva'}, 'GRAMATICA': {'TIPO': 'adverbio_estado', 'REQUIERE': ['verbo_estado'], 'PERMITE_DESPUES': ['pregunta_reciproca', 'fin_frase']}}),
        ("gen_genial", "genial", {'idioma': 'es', 'semantic_field': 'respuesta_bienestar', 'personalidad': {'tono': 'positivo', 'emocion': 'positiva'}, 'GRAMATICA': {'TIPO': 'adverbio_estado', 'PERMITE_DESPUES': ['pregunta_reciproca', 'fin_frase']}}),
        ("gen_muy", "muy", {'idioma': 'es', 'semantic_field': 'intensificador', 'personalidad': {'tono': 'neutral'}, 'GRAMATICA': {'TIPO': 'adverbio_intensidad', 'REQUIERE': ['verbo_estado'], 'PERMITE_DESPUES': ['adverbio_estado', 'adjetivo']}}),
        ("gen_servicial", "servicial", {'idioma': 'es', 'semantic_field': 'rasgo_positivo', 'personalidad': {'tono': 'positivo', 'emocion': 'positiva'}, 'GRAMATICA': {'TIPO': 'adjetivo_calificativo', 'REQUIERE': ['sustantivo', 'verbo_estado'], 'PERMITE_DESPUES': ['fin_frase']}}),
        ("gen_de_nada", "de nada", {'idioma': 'es', 'semantic_field': 'respuesta_agradecimiento', 'personalidad': {'tono': 'amable', 'emocion': 'positiva'}, 'GRAMATICA': {'TIPO': 'cortesia_respuesta', 'REQUIERE': ['inicio_frase'], 'PERMITE_DESPUES': ['fin_frase']}}),
        ("gen_pues", "pues", {'idioma': 'es', 'semantic_field': 'conector_discursivo', 'GRAMATICA': {'TIPO': 'conector', 'REQUIERE': ['inicio_frase'], 'PERMITE_DESPUES': ['pronombre']}}),
        ("gen_en_que", "en qué", {'idioma': 'es', 'semantic_field': 'pregunta_especifica', 'GRAMATICA': {'TIPO': 'interrogativo_complejo', 'REQUIERE': ['inicio_frase'], 'PERMITE_DESPUES': ['verbo_modal']}}),
        ("gen_fin", "<FIN>", {'GRAMATICA': {'TIPO': 'fin_frase'}}),  # sin idioma

        # ========== INGLÉS ==========
        # --- Saludos y despedidas ---
        ("mn_hello", "hello", {'idioma': 'en', 'semantic_field': 'saludo', 'personalidad': {'tono': 'casual', 'emocion': 'neutral'}, 'GRAMATICA': {'TIPO': 'saludo', 'PERMITE_DESPUES': ['interrogativo_estado', 'fin_frase']}}),
        ("mn_hi", "hi", {'idioma': 'en', 'semantic_field': 'saludo', 'personalidad': {'tono': 'casual', 'emocion': 'neutral'}, 'GRAMATICA': {'TIPO': 'saludo_informal'}}),
        ("mn_good_morning", "good morning", {'idioma': 'en', 'GRAMATICA': {'TIPO': 'saludo_formal'}}),
        ("mn_good_afternoon", "good afternoon", {'idioma': 'en', 'GRAMATICA': {'TIPO': 'saludo_formal'}}),
        ("mn_good_evening", "good evening", {'idioma': 'en', 'GRAMATICA': {'TIPO': 'saludo_formal'}}),
        ("mn_bye", "bye", {'idioma': 'en', 'semantic_field': 'despedida', 'personalidad': {'tono': 'casual', 'emocion': 'neutral'}, 'GRAMATICA': {'TIPO': 'despedida_informal'}}),
        ("mn_goodbye", "goodbye", {'idioma': 'en', 'semantic_field': 'despedida', 'personalidad': {'tono': 'neutral', 'emocion': 'neutral'}, 'GRAMATICA': {'TIPO': 'despedida', 'PERMITE_DESPUES': ['fin_frase']}}),
        ("mn_see_you", "see you", {'idioma': 'en', 'semantic_field': 'despedida', 'GRAMATICA': {'TIPO': 'despedida_compuesta'}}),

        # --- Interrogativos ---
        ("mn_what", "what", {'idioma': 'en', 'semantic_field': 'pregunta_general', 'personalidad': {'tono': 'neutral'}, 'GRAMATICA': {'TIPO': 'interrogativo_general', 'PERMITE_DESPUES': ['sustantivo', 'verbo']}}),
        ("mn_who", "who", {'idioma': 'en', 'semantic_field': 'pregunta_identidad', 'GRAMATICA': {'TIPO': 'interrogativo_persona'}}),
        ("mn_how", "how", {'idioma': 'en', 'semantic_field': 'pregunta_bienestar', 'personalidad': {'tono': 'neutral'}, 'GRAMATICA': {'TIPO': 'interrogativo_estado', 'PERMITE_DESPUES': ['verbo_estado']}}),
        ("mn_when", "when", {'idioma': 'en', 'semantic_field': 'pregunta_tiempo', 'GRAMATICA': {'TIPO': 'interrogativo_temporal'}}),
        ("mn_where", "where", {'idioma': 'en', 'semantic_field': 'pregunta_lugar', 'GRAMATICA': {'TIPO': 'interrogativo_lugar'}}),
        ("mn_why", "why", {'idioma': 'en', 'semantic_field': 'pregunta_razon', 'GRAMATICA': {'TIPO': 'interrogativo_causal'}}),
        ("mn_which", "which", {'idioma': 'en', 'semantic_field': 'pregunta_especifica', 'GRAMATICA': {'TIPO': 'interrogativo_seleccion'}}),

        # --- Verbos ---
        ("mn_be", "be", {'idioma': 'en', 'GRAMATICA': {'TIPO': 'verbo_infinitivo', 'RAIZ': 'be'}}),
        ("mn_am", "am", {'idioma': 'en', 'GRAMATICA': {'TIPO': 'verbo_estado', 'PERSONA': 1, 'TIEMPO': 'presente'}}),
        ("mn_is", "is", {'idioma': 'en', 'GRAMATICA': {'TIPO': 'verbo_estado', 'PERSONA': 3, 'TIEMPO': 'presente'}}),
        ("mn_are", "are", {'idioma': 'en', 'GRAMATICA': {'TIPO': 'verbo_estado', 'PERSONA': 2, 'TIEMPO': 'presente'}}),
        ("mn_was", "was", {'idioma': 'en', 'GRAMATICA': {'TIPO': 'verbo_estado', 'TIEMPO': 'pasado'}}),
        ("mn_were", "were", {'idioma': 'en', 'GRAMATICA': {'TIPO': 'verbo_estado', 'TIEMPO': 'pasado'}}),
        ("mn_have", "have", {'idioma': 'en', 'GRAMATICA': {'TIPO': 'verbo_posesion', 'PERSONA': 1, 'TIEMPO': 'presente'}}),
        ("mn_has", "has", {'idioma': 'en', 'GRAMATICA': {'TIPO': 'verbo_posesion', 'PERSONA': 3, 'TIEMPO': 'presente'}}),
        ("mn_do", "do", {'idioma': 'en', 'GRAMATICA': {'TIPO': 'verbo_accion', 'TIEMPO': 'presente'}}),
        ("mn_does", "does", {'idioma': 'en', 'GRAMATICA': {'TIPO': 'verbo_accion', 'PERSONA': 3, 'TIEMPO': 'presente'}}),
        ("mn_can", "can", {'idioma': 'en', 'semantic_field': 'auto_revelacion_capacidad', 'personalidad': {'tono': 'neutral'}, 'GRAMATICA': {'TIPO': 'verbo_modal', 'PERMITE_DESPUES': ['verbo_infinitivo']}}),
        ("mn_want", "want", {'idioma': 'en', 'GRAMATICA': {'TIPO': 'verbo_deseo', 'TIEMPO': 'presente'}}),

        # --- Afirmación / Negación / Cortesía ---
        ("mn_yes", "yes", {'idioma': 'en', 'semantic_field': 'afirmacion', 'GRAMATICA': {'TIPO': 'afirmacion'}}),
        ("mn_no_en", "no", {'idioma': 'en', 'semantic_field': 'negacion', 'GRAMATICA': {'TIPO': 'negacion'}}),
        ("mn_thanks", "thanks", {'idioma': 'en', 'semantic_field': 'agradecimiento', 'personalidad': {'tono': 'positivo', 'emocion': 'positiva'}, 'GRAMATICA': {'TIPO': 'cortesia', 'PERMITE_DESPUES': ['fin_frase']}}),
        ("mn_thank_you", "thank you", {'idioma': 'en', 'semantic_field': 'agradecimiento', 'personalidad': {'tono': 'positivo', 'emocion': 'positiva'}, 'GRAMATICA': {'TIPO': 'cortesia', 'PERMITE_DESPUES': ['fin_frase']}}),
        ("mn_youre_welcome", "you're welcome", {'idioma': 'en', 'semantic_field': 'respuesta_agradecimiento', 'personalidad': {'tono': 'amable', 'emocion': 'positiva'}, 'GRAMATICA': {'TIPO': 'cortesia_respuesta', 'REQUIERE': ['inicio_frase'], 'PERMITE_DESPUES': ['fin_frase']}}),

        # --- Generales ---
        ("mn_name", "name", {'idioma': 'en', 'semantic_field': 'identidad', 'GRAMATICA': {'TIPO': 'sustantivo'}}),
        ("mn_time", "time", {'idioma': 'en', 'semantic_field': 'temporalidad', 'GRAMATICA': {'TIPO': 'sustantivo'}}),
        ("mn_weather", "weather", {'idioma': 'en', 'semantic_field': 'ambiente', 'GRAMATICA': {'TIPO': 'sustantivo'}}),
        ("mn_help", "help", {'idioma': 'en', 'semantic_field': 'asistencia', 'GRAMATICA': {'TIPO': 'sustantivo_o_verbo'}}),
        ("mn_you", "you", {'idioma': 'en', 'semantic_field': 'identidad_externa', 'GRAMATICA': {'TIPO': 'pronombre_personal'}}),
        ("mn_i", "I", {'idioma': 'en', 'semantic_field': 'identidad_propia', 'personalidad': {'tono': 'neutral'}, 'GRAMATICA': {'TIPO': 'pronombre', 'PERMITE_DESPUES': ['verbo_estado', 'verbo_accion']}}),

        # --- Matemáticas (inglés) ---
        ("mn_zero", "zero", {'idioma': 'en', 'semantic_field': 'numero', 'valor': 0}),
        ("mn_one", "one", {'idioma': 'en', 'semantic_field': 'numero', 'valor': 1}),
        ("mn_two", "two", {'idioma': 'en', 'semantic_field': 'numero', 'valor': 2}),
        ("mn_three", "three", {'idioma': 'en', 'semantic_field': 'numero', 'valor': 3}),
        ("mn_four", "four", {'idioma': 'en', 'semantic_field': 'numero', 'valor': 4}),
        ("mn_five", "five", {'idioma': 'en', 'semantic_field': 'numero', 'valor': 5}),
        ("mn_six", "six", {'idioma': 'en', 'semantic_field': 'numero', 'valor': 6}),
        ("mn_seven", "seven", {'idioma': 'en', 'semantic_field': 'numero', 'valor': 7}),
        ("mn_eight", "eight", {'idioma': 'en', 'semantic_field': 'numero', 'valor': 8}),
        ("mn_nine", "nine", {'idioma': 'en', 'semantic_field': 'numero', 'valor': 9}),
        ("mn_ten", "ten", {'idioma': 'en', 'semantic_field': 'numero', 'valor': 10}),
        ("mn_plus", "plus", {'idioma': 'en', 'semantic_field': 'operacion_suma', 'simbolo': '+', 'GRAMATICA': {'TIPO': 'operador'}}),
        ("mn_minus", "minus", {'idioma': 'en', 'semantic_field': 'operacion_resta', 'simbolo': '-', 'GRAMATICA': {'TIPO': 'operador'}}),
        ("mn_times", "times", {'idioma': 'en', 'semantic_field': 'operacion_multiplicacion', 'simbolo': '*', 'GRAMATICA': {'TIPO': 'operador'}}),
        ("mn_divided_by", "divided by", {'idioma': 'en', 'semantic_field': 'operacion_division', 'simbolo': '/', 'GRAMATICA': {'TIPO': 'operador'}}),
        ("mn_how_much_is", "how much is", {'idioma': 'en', 'semantic_field': 'pregunta_calculo', 'GRAMATICA': {'TIPO': 'pregunta_calculo'}}),
        ("mn_result", "result", {'idioma': 'en', 'semantic_field': 'respuesta_calculo', 'GRAMATICA': {'TIPO': 'sustantivo'}}),

        # --- Palabras de clarificación (inglés) - NUEVAS ---
        ("mn_clarify", "clarify", {'idioma': 'en', 'semantic_field': 'clarificacion', 'personalidad': {'tono': 'neutral'}, 'GRAMATICA': {'TIPO': 'verbo'}}),
        ("mn_explain", "explain", {'idioma': 'en', 'semantic_field': 'clarificacion', 'personalidad': {'tono': 'neutral'}, 'GRAMATICA': {'TIPO': 'verbo'}}),

        # --- Generativas (inglés) ---
        ("gen_and_you", "and you?", {'idioma': 'en', 'semantic_field': 'pregunta_reciproca', 'personalidad': {'tono': 'casual', 'emocion': 'curiosidad'}, 'GRAMATICA': {'TIPO': 'pregunta_reciproca', 'REQUIERE': ['adverbio_estado'], 'PERMITE_DESPUES': ['fin_frase']}}),
        ("gen_help_you", "help you", {'idioma': 'en', 'semantic_field': 'asistencia', 'personalidad': {'tono': 'proactivo', 'emocion': 'positiva'}, 'GRAMATICA': {'TIPO': 'verbo_infinitivo_complejo', 'REQUIERE': ['verbo_modal'], 'PERMITE_DESPUES': ['fin_frase']}}),
        ("gen_analyze", "analyze", {'idioma': 'en', 'semantic_field': 'auto_revelacion_capacidad', 'GRAMATICA': {'TIPO': 'verbo_infinitivo'}}),
        ("gen_learn", "learn", {'idioma': 'en', 'semantic_field': 'auto_revelacion_capacidad', 'GRAMATICA': {'TIPO': 'verbo_infinitivo'}}),
        ("gen_process", "process", {'idioma': 'en', 'semantic_field': 'auto_revelacion_capacidad', 'GRAMATICA': {'TIPO': 'verbo_infinitivo'}}),
        ("gen_ai", "an AI", {'idioma': 'en', 'semantic_field': 'identidad_propia', 'personalidad': {'tono': 'neutral'}, 'GRAMATICA': {'TIPO': 'sustantivo', 'REQUIERE': ['verbo_estado'], 'PERMITE_DESPUES': ['adjetivo_calificativo', 'fin_frase']}}),
        ("gen_text", "text", {'idioma': 'en', 'semantic_field': 'auto_revelacion_capacidad', 'GRAMATICA': {'TIPO': 'sustantivo'}}),
        ("gen_language", "language", {'idioma': 'en', 'semantic_field': 'auto_revelacion_capacidad', 'GRAMATICA': {'TIPO': 'sustantivo'}}),
        ("gen_good", "good", {'idioma': 'en', 'semantic_field': 'respuesta_bienestar', 'personalidad': {'tono': 'positivo', 'emocion': 'positiva'}, 'GRAMATICA': {'TIPO': 'adverbio_estado', 'REQUIERE': ['verbo_estado'], 'PERMITE_DESPUES': ['pregunta_reciproca', 'fin_frase']}}),
        ("gen_great", "great", {'idioma': 'en', 'semantic_field': 'respuesta_bienestar', 'personalidad': {'tono': 'positivo', 'emocion': 'positiva'}, 'GRAMATICA': {'TIPO': 'adverbio_estado', 'PERMITE_DESPUES': ['pregunta_reciproca', 'fin_frase']}}),
        ("gen_very", "very", {'idioma': 'en', 'semantic_field': 'intensificador', 'personalidad': {'tono': 'neutral'}, 'GRAMATICA': {'TIPO': 'adverbio_intensidad', 'REQUIERE': ['verbo_estado'], 'PERMITE_DESPUES': ['adverbio_estado', 'adjetivo']}}),
        ("gen_helpful", "helpful", {'idioma': 'en', 'semantic_field': 'rasgo_positivo', 'personalidad': {'tono': 'positivo', 'emocion': 'positiva'}, 'GRAMATICA': {'TIPO': 'adjetivo_calificativo', 'REQUIERE': ['sustantivo', 'verbo_estado'], 'PERMITE_DESPUES': ['fin_frase']}}),
        ("gen_well", "well", {'idioma': 'en', 'semantic_field': 'respuesta_bienestar', 'personalidad': {'tono': 'positivo', 'emocion': 'positiva'}, 'GRAMATICA': {'TIPO': 'adverbio_estado', 'REQUIERE': ['verbo_estado'], 'PERMITE_DESPUES': ['pregunta_reciproca', 'fin_frase']}}),
        ("gen_so", "so", {'idioma': 'en', 'semantic_field': 'conector_discursivo', 'GRAMATICA': {'TIPO': 'conector', 'REQUIERE': ['inicio_frase'], 'PERMITE_DESPUES': ['pronombre']}}),
        ("gen_in_what", "in what", {'idioma': 'en', 'semantic_field': 'pregunta_especifica', 'GRAMATICA': {'TIPO': 'interrogativo_complejo', 'REQUIERE': ['inicio_frase'], 'PERMITE_DESPUES': ['verbo_modal']}}),
        # FIN ya está definido antes, no duplicamos
    ]

    # ---- Micro‑neuronas abstractas (conceptos) ----
    abstract_concepts = [
        ("concepto_saludo", "la idea de un saludo"),
        ("concepto_despedida", "la idea de una despedida"),
        ("concepto_pregunta_bienestar", "la idea de preguntar como esta alguien"),
        ("concepto_respuesta_bienestar", "la idea de afirmar el propio bienestar"),
        ("concepto_pregunta_reciproca", "la idea de devolver una pregunta social"),
        ("concepto_agradecimiento", "la idea de dar las gracias"),
        ("concepto_respuesta_agradecimiento", "la idea de responder a un gracias"),
        ("concepto_peticion_info_general", "la idea de pedir información general"),
        ("concepto_peticion_info_especifica", "la idea de pedir información sobre un tema"),
        ("concepto_ofrecer_ayuda", "la idea de ofrecer asistencia"),
        ("concepto_pedir_ayuda", "la idea de solicitar ayuda"),
        ("concepto_auto_revelacion_identidad", "la idea de revelar quién soy"),
        ("concepto_auto_revelacion_capacidad", "la idea de revelar qué puedo hacer"),
        ("concepto_acuerdo", "la idea de estar de acuerdo"),
        ("concepto_desacuerdo", "la idea de no estar de acuerdo"),
        ("concepto_empatia_positiva", "la idea de compartir la alegría de alguien"),
        ("concepto_empatia_negativa", "la idea de mostrar comprensión ante el malestar"),
        ("concepto_clarificacion", "la idea de pedir que se aclare algo"),
        ("concepto_iniciar_conversacion", "la idea de empezar a hablar proactivamente"),
        # Nuevos conceptos matemáticos
        ("concepto_numero", "la idea de un número"),
        ("concepto_operacion_suma", "la idea de sumar"),
        ("concepto_operacion_resta", "la idea de restar"),
        ("concepto_operacion_multiplicacion", "la idea de multiplicar"),
        ("concepto_operacion_division", "la idea de dividir"),
        ("concepto_pregunta_calculo", "la idea de preguntar por un cálculo"),
        ("concepto_resultado_calculo", "la idea del resultado de un cálculo"),
        ("concepto_tiempo", "la idea del tiempo"),
        ("concepto_clima", "la idea del clima"),
        ("concepto_identidad_externa", "la idea de la otra persona"),
        ("concepto_pregunta_identidad", "la idea de preguntar quién es alguien"),
        ("concepto_pregunta_tiempo", "la idea de preguntar la hora"),
        ("concepto_pregunta_lugar", "la idea de preguntar por un lugar"),
        ("concepto_pregunta_razon", "la idea de preguntar por una razón"),
    ]

    # ---- Neuronas (patrones) ----
    neurons = [
        ("n_patron_saludo_hola", "Patrón: Hola", ["mn_hola"], []),
        ("n_patron_saludo_formal", "Patrón: Buenos días/tardes/noches", ["mn_buenos", "mn_dias"], []),
        ("n_patron_despedida", "Patrón: Adiós/Chao/etc", ["mn_adios"], []),
        ("n_pregunta_como_estas", "Patrón: ¿Cómo estás?", ["mn_como", "mn_estas"], ["mn_no"]),
        ("n_saludo_que_tal", "Patrón: ¿Qué tal?", ["mn_que", "mn_tal"], []),
        ("n_pregunta_quien_eres", "Patrón: ¿Quién eres?", ["mn_quien", "mn_eres"], []),
        ("n_pregunta_que_haces", "Patrón: ¿Qué haces?", ["mn_que", "mn_haces"], ["mn_tal"]),
        ("n_peticion_ayuda", "Patrón: ¿Puedes ayudar?", ["mn_puedes", "mn_ayuda"], []),
        ("n_peticion_nombre", "Patrón: Pregunta por nombre", ["mn_cual", "mn_es", "mn_tu", "mn_nombre"], []),
        ("n_agradecimiento", "Patrón: Usuario da las gracias", ["mn_gracias"], []),
        ("n_afirmacion_positiva", "Patrón: Usuario afirma 'si/claro/ok'", ["mn_si"], []),
        ("n_negacion", "Patrón: Usuario dice 'no'", ["mn_no"], []),
        ("n_charla_sobre_clima", "Patrón: Usuario menciona el clima", ["mn_clima"], []),
        # Nuevos patrones para cálculos
        ("n_pregunta_calculo_simple", "Patrón: Pregunta de cálculo (ej. cuánto es 2+2)", ["mn_cuanto_es", "mn_dos", "mn_mas", "mn_dos"], []),
        ("n_pregunta_calculo_ingles", "Pattern: Math question (how much is 2+2)", ["mn_how_much_is", "mn_two", "mn_plus", "mn_two"], []),
    ]

    # Extraer listas para micro‑neuronas
    micro_ids = []
    micro_concepts = []
    micro_types = []
    micro_metadata = []


    for mid, concept, meta in unified_vocabulary:
        meta = meta.copy()
        field = meta.get('semantic_field')
        if field in field_to_concept and field_to_concept[field] is not None:
            meta['concept_id'] = field_to_concept[field]
        else:
            meta['concept_id'] = ''
        micro_ids.append(mid)
        micro_concepts.append(concept)
        micro_types.append('keyword')
        micro_metadata.append(meta)

    for mid, concept in abstract_concepts:
        meta = {}
        meta['concept_id'] = mid
        micro_ids.append(mid)
        micro_concepts.append(concept)
        micro_types.append('abstract_concept')
        micro_metadata.append(meta)

    # Extraer listas para neuronas
    neuron_ids = []
    neuron_names = []
    neuron_conditions = []
    neuron_exclusions = []
    for nid, name, conds, excs in neurons:
        neuron_ids.append(nid)
        neuron_names.append(name)
        neuron_conditions.append(conds)
        neuron_exclusions.append(excs)

    # ---- Macro‑neuronas (conceptos) ----
    # Se crean a partir de los conceptos abstractos (abstract_concepts)
    macro_ids = [mid for mid, _ in abstract_concepts]
    macro_names = [name for _, name in abstract_concepts]
    # Para cada concepto, sus condiciones y exclusiones (vacías por ahora)
    macro_conditions = [[] for _ in macro_ids]
    macro_exclusions = [[] for _ in macro_ids]
    macro_metadata = [{} for _ in macro_ids]

    return {
        'micro_ids': micro_ids,
        'micro_concepts': micro_concepts,
        'micro_types': micro_types,
        'micro_metadata': micro_metadata,
        'neuron_ids': neuron_ids,
        'neuron_names': neuron_names,
        'neuron_conditions': neuron_conditions,
        'neuron_exclusions': neuron_exclusions,
        'macro_ids': macro_ids,
        'macro_names': macro_names,
        'macro_conditions': macro_conditions,
        'macro_exclusions': macro_exclusions,
        'macro_metadata': macro_metadata,
    }