# deliberation -> response_builder.py

import torch
import time
import logging
from typing import List, Dict, Any, Optional

from ..states.interconnector_state import InterconnectorState
from ..language.grammar_adjudicator import GrammarAdjudicator
from ..core.personality import Personality
from ..core.cognitive_engine import CognitiveEngine


class ResponseBuilder:
    """
    Genera respuestas en lenguaje natural a partir de un plan conceptual.
    Soporta plantillas, evaluación de expresiones matemáticas y registro de operaciones desconocidas.
    """
    def __init__(self,
                 personality: Personality,
                 context: Any,
                 winning_plan: Optional[Dict],
                 adjudicator: GrammarAdjudicator,
                 engine: CognitiveEngine,
                 neural_system=None,
                 interconnectors: Optional[InterconnectorState] = None):
        self.personality = personality
        self.context = context
        self.winning_plan = winning_plan
        self.adjudicator = adjudicator
        self.engine = engine
        self.neural_system = neural_system
        self.interconnectors = interconnectors
        self.language = 'es'  # valor por defecto (se puede cambiar externamente)

        # Logger
        self.logger = logging.getLogger(__name__)

        # Diccionario: concept_id (macro) -> lista de ids de micro‑neuronas (palabras) asociadas
        self.concept_to_words: Dict[str, List[str]] = {}
        # Diccionario: micro_id -> language
        self.word_language: Dict[str, str] = {}

        # Mapeo ID -> índice para micro‑neuronas
        self.micro_id_to_idx: Dict[str, int] = {}
        if self.engine.micro_state is not None:
            for i, id_ in enumerate(self.engine.micro_state.ids):
                self.micro_id_to_idx[id_] = i

        # Construir mapas iniciales
        self._update_maps_from_micro()
        self.logger.debug(f"[ResponseBuilder] concept_to_words keys: {list(self.concept_to_words.keys())}")
        self.logger.debug(f"[ResponseBuilder] word_language items: {list(self.word_language.items())[:10]}")
        # --- Mapeos para evaluación de expresiones ---
        # Valor numérico de cada micro (si es un número)
        self.number_value = {
            # Español
            'mn_cero': 0, 'mn_uno': 1, 'mn_dos': 2, 'mn_tres': 3, 'mn_cuatro': 4,
            'mn_cinco': 5, 'mn_seis': 6, 'mn_siete': 7, 'mn_ocho': 8, 'mn_nueve': 9, 'mn_diez': 10,
            # Inglés
            'mn_zero': 0, 'mn_one': 1, 'mn_two': 2, 'mn_three': 3, 'mn_four': 4,
            'mn_five': 5, 'mn_six': 6, 'mn_seven': 7, 'mn_eight': 8, 'mn_nine': 9, 'mn_ten': 10,
        }
        # Mapeo de operadores a símbolos (para _evaluate_expression)
        self.supported_operators = {
            'mn_mas': '+', 'mn_plus': '+',
            'mn_menos': '-', 'mn_minus': '-',
            'mn_por': '*', 'mn_times': '*',
            'mn_dividido': '/', 'mn_divided_by': '/',
            'mn_modulo': '%', 'mn_mod': '%'
        }

        # Mapeo de número a palabra para resultados simples (0-10)
        if self.language == 'es':
            self.number_to_word = {i: word for i, word in enumerate(
                ['cero', 'uno', 'dos', 'tres', 'cuatro', 'cinco', 'seis', 'siete', 'ocho', 'nueve', 'diez'])}
        else:  # inglés
            self.number_to_word = {i: word for i, word in enumerate(
                ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'])}

        self.logger.debug(f"Concept_to_words: {list(self.concept_to_words.keys())}")
        self.logger.debug(f"Word_language: {list(self.word_language.items())[:5]}")

    # ------------------------------------------------------------------
    # Métodos públicos
    # ------------------------------------------------------------------
    def add_concept(self, concept_id: str, word_ids: List[str]):
        """
        Añade un nuevo concepto y las palabras asociadas a los mapas internos.
        Se llama después de crear un nuevo concepto en caliente (clustering).
        """
        if concept_id not in self.concept_to_words:
            self.concept_to_words[concept_id] = []
        self.concept_to_words[concept_id].extend(word_ids)

        # Actualizar word_language si es necesario
        for mid in word_ids:
            if mid in self.micro_id_to_idx:
                idx = self.micro_id_to_idx[mid]
                meta = self.engine.micro_state.metadata[idx]
                if 'idioma' in meta:
                    self.word_language[mid] = meta['idioma']
        self.logger.info(f"Concepto {concept_id} añadido con palabras: {word_ids}")

    def build_response(self) -> str:
        """Punto de entrada principal: genera la respuesta a partir del plan ganador."""
        self.logger.debug("Iniciando construcción de respuesta")
        if not self.winning_plan:
            return "No estoy segura de qué decir."
        return self._synthesize_emergent_response(self.winning_plan)

    # ------------------------------------------------------------------
    # Métodos privados de soporte
    # ------------------------------------------------------------------
    def _update_maps_from_micro(self):
        """(Re)construye los mapas concept_to_words y word_language desde micro_state."""
        self.concept_to_words.clear()
        self.word_language.clear()
        if self.engine.micro_state is not None:
            micro = self.engine.micro_state
            for i, mid in enumerate(micro.ids):
                meta = micro.metadata[i]
                concept_id = meta.get('concept_id')
                if concept_id:
                    if concept_id not in self.concept_to_words:
                        self.concept_to_words[concept_id] = []
                    self.concept_to_words[concept_id].append(mid)
                language = meta.get('language')
                if language:
                    self.word_language[mid] = language

    def _get_micro_metadata(self, micro_id: str) -> Dict:
        idx = self.micro_id_to_idx.get(micro_id)
        if idx is None:
            return {}
        return self.engine.micro_state.metadata[idx]

    # ------------------------------------------------------------------
    # Evaluación de expresiones matemáticas
    # ------------------------------------------------------------------
    def _evaluate_expression(self, number_ids: List[str], operator_ids: List[str]) -> str:
        """
        Evalúa una expresión matemática representada por IDs de micro-neuronas.
        Retorna un string con el resultado o un mensaje de error/desconocido.
        """
        # Verificar operadores conocidos
        for op_id in operator_ids:
            if op_id not in self.supported_operators:
                self._log_unknown_operation(f"operador {op_id}", "operator_not_supported")
                return f"No sé hacer esa operación ({op_id}) aún. Estoy aprendiendo."

        # Si es expresión compleja (más de 2 números o más de 1 operador)
        if len(number_ids) > 2 or len(operator_ids) > 1:
            self._log_unknown_operation(f"expresión compleja: nums={number_ids}, ops={operator_ids}",
                                                   "complex_expression")
            return "Expresiones complejas están en desarrollo. Por ahora solo acepto operaciones simples (ej. 2+2)."

        # Caso simple: dos números y un operador
        if len(number_ids) == 2 and len(operator_ids) == 1:
            a = self.number_value.get(number_ids[0])
            b = self.number_value.get(number_ids[1])
            if a is None or b is None:
                return "Uno de los números no lo reconozco."

            op = self.supported_operators[operator_ids[0]]
            try:
                if op == '+':
                    res = a + b
                elif op == '-':
                    res = a - b
                elif op == '*':
                    res = a * b
                elif op == '/':
                    if b == 0:
                        return "No puedo dividir entre cero."
                    res = a / b
                elif op == '%':
                    res = a % b
                else:
                    return "Operación no soportada."
            except Exception as e:
                self.logger.error(f"Error evaluando expresión: {e}")
                return f"Error al calcular: {e}"

            # Convertir resultado a palabra si es entero 0-10
            if isinstance(res, (int, float)) and res.is_integer() and 0 <= int(res) <= 10:
                return self.number_to_word[int(res)]
            else:
                return str(res)

        # Si no es el caso simple (no debería llegar aquí por las comprobaciones anteriores)
        return "No puedo evaluar esa expresión todavía. Estoy aprendiendo."

    def _log_unknown_operation(self, expression: str, type: str = "operator_not_supported"):
        """
        Registra una operación no resuelta para depuración o aprendizaje futuro.
        """
        self.logger.info(f"OPERACIÓN DESCONOCIDA: {expression} (tipo={type})")
        if hasattr(self.engine, 'memory') and self.engine.memory:
            self.engine.memory.add_to_memory({
                'type': 'unknown_operation',
                'expression': expression,
                'timestamp': time.time()
            }, pool='aprendizaje')

    # ------------------------------------------------------------------
    # Generación de frases (plantillas + fallback)
    # ------------------------------------------------------------------
    def _generate_phrase(self, conceptual_plan: List[str], metadata: Dict) -> str:
        """
        Genera una frase a partir del plan conceptual.
        Primero busca plantillas, si no, concatena palabras elegidas.
        """
        # Plantillas básicas (ampliable)
        templates = {
            ('concepto_saludo', 'concepto_pregunta_bienestar'): {
                'es': "¡Hola! ¿Cómo estás?",
                'en': "Hello! How are you?"
            },
            ('concepto_saludo',): {
                'es': "¡Hola!",
                'en': "Hello!"
            },
            ('concepto_respuesta_bienestar',): {
                'es': "Estoy bien, gracias.",
                'en': "I'm fine, thanks."
            }
        }
        key = tuple(conceptual_plan)
        if key in templates:
            return templates[key].get(self.language, templates[key]['es'])

        # Fallback: concatenar palabras
        words = []
        for concept in conceptual_plan:
            word = self._choose_word_for_concept(concept)
            if word:
                words.append(word)
        return ' '.join(words)

    def _choose_word_for_concept(self, concept_id: str) -> Optional[str]:
        """
        Devuelve una palabra (texto) adecuada para el concepto, según idioma.
        """
        word_ids = self.concept_to_words.get(concept_id, [])
        if not word_ids:
            return None

        # Filtrar por idioma si es posible
        for wid in word_ids:
            if self.word_language.get(wid) == self.language:
                idx = self.micro_id_to_idx.get(wid)
                if idx is not None:
                    return self.engine.micro_state.concepts[idx]
        # Si no hay del idioma, usar la primera
        first_id = word_ids[0]
        idx = self.micro_id_to_idx.get(first_id)
        if idx is not None:
            return self.engine.micro_state.concepts[idx]
        return first_id  # fallback extremo

    # ------------------------------------------------------------------
    # Síntesis principal (orquesta todo)
    # ------------------------------------------------------------------
    def _synthesize_emergent_response(self, plan: Dict) -> str:
        """
        Método principal que decide cómo generar la respuesta según el plan.
        """
        logging.getLogger().debug("Iniciando síntesis de respuesta")
        conceptual_plan = plan.get("conceptual_plan")
        if not conceptual_plan:
            return "Mi pensamiento no está claro."

        self.logger.debug(f"Plan conceptual recibido: {conceptual_plan}")

        # --- CASO ESPECIAL: operación matemática ---
        if conceptual_plan == ['concepto_resultado_calculo']:
            metadata = plan.get('metadata', {})
            numbers = metadata.get('numeros', [])
            operators = metadata.get('operadores', [])
            if numbers and operators:
                result_str = self._evaluate_expression(numbers, operators)
                # Si el resultado es un mensaje de error/desconocido, ya lo hemos registrado dentro
                return result_str
            else:
                # Si no hay metadata, caemos en generación normal (por si acaso)
                self.logger.warning("Plan de cálculo sin metadata, usando fallback")
                pass

        # --- OTROS PLANES: usar generación por plantillas/concatenación ---
        return self._generate_phrase(conceptual_plan, plan.get('metadata', {})) 