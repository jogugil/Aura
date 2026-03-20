# thinking_neurons.py - Versión unificada sin dependencia circular
# Contiene tanto las Thinking Neurons como el MacroTN

import collections
import logging
from typing import List, Dict, Any, Optional

# Configurar logger
logger = logging.getLogger(__name__)


# --- 1. Thought Proposal Data Structure ---
class ThoughtProposal:
    def __init__(self, source, conceptual_plan, confidence, reasoning, metadata=None):
        self.source = source.__class__.__name__
        self.conceptual_plan = conceptual_plan
        self.confidence = confidence
        self.reasoning = reasoning
        self.metadata = metadata if metadata is not None else {}


# --- 2. Base Class for Thinking Neurons ---
class BaseThinkingNeuron:
    """Clase base abstracta. Las dependencias se inyectan externamente."""
    def __init__(self):
        if self.__class__ == BaseThinkingNeuron:
            raise TypeError("No se puede instanciar BaseThinkingNeuron directamente.")
        self.name = self.__class__.__name__
        # Dependencias inyectables
        self.engine = None
        self.memory = None
        self.neural_system = None
        self.interconnectors = None

    def propose(self, neural_state, retrieved_memories, context_hypotheses):
        raise NotImplementedError


# --- 3. Concrete Thinking Neurons ---
class SocialProtocolTN(BaseThinkingNeuron):
    """Punto de vista social: respuestas corteses y esperadas."""
    
    def _concept_exists(self, concept_id: str) -> bool:
        """Verifica si un concepto existe en macro_state."""
        if self.engine and self.engine.macro_state:
            return concept_id in self.engine.macro_state.ids
        return False

    def propose(self, neural_state, retrieved_memories, context_hypotheses):
        logger.debug(f"[DEBUG TN SocialProtocol] Evaluando {len(context_hypotheses)} hipótesis")
        for i, scenario in enumerate(context_hypotheses):
            hypo_type = scenario.get('type')
            logger.debug(f"[DEBUG TN SocialProtocol] Hipótesis {i}: type={hypo_type}, subtype={scenario.get('subtype')}, confidence={scenario.get('confidence')}")
            
            # --- NUEVO: responder a palabras sueltas (thinking_memory_focus) ---
            if hypo_type == 'thinking_memory_focus':
                key_elements = scenario.get('key_elements', [])
                if key_elements and len(key_elements) == 1:
                    word_id = key_elements[0]
                    # Buscar si la palabra es un saludo conocido
                    if word_id in ['mn_hola', 'mn_hello', 'mn_hi', 'mn_buenos', 'mn_dias']:
                        if self._concept_exists('concepto_saludo') and self._concept_exists('concepto_pregunta_bienestar'):
                            plan = ['concepto_saludo', 'concepto_pregunta_bienestar']
                            reasoning = f"Saludo detectado: {word_id}"
                            conf = scenario.get('confidence', 0.8)
                            logger.debug(f"[DEBUG TN SocialProtocol] Propuesta generada por palabra suelta: {plan}")
                            return ThoughtProposal(self, plan, conf, reasoning)
                    elif word_id in ['mn_adios', 'mn_bye', 'mn_hasta_luego']:
                        if self._concept_exists('concepto_despedida'):
                            plan = ['concepto_despedida']
                            reasoning = f"Despedida detectada: {word_id}"
                            conf = scenario.get('confidence', 0.8)
                            return ThoughtProposal(self, plan, conf, reasoning)
                    elif word_id in ['mn_gracias', 'mn_thanks']:
                        if self._concept_exists('concepto_respuesta_agradecimiento'):
                            plan = ['concepto_respuesta_agradecimiento']
                            reasoning = f"Agradecimiento detectado: {word_id}"
                            conf = scenario.get('confidence', 0.8)
                            return ThoughtProposal(self, plan, conf, reasoning)

            # Patrón aprendido de saludo (original)
            if hypo_type == 'learned_pattern':
                evidence_support = scenario.get('evidence_support', {})
                if "saludo" in evidence_support.get('pattern', []):
                    conf = scenario.get('confidence', 0.8)
                    if (self._concept_exists('concepto_saludo') and 
                        self._concept_exists('concepto_pregunta_bienestar')):
                        plan = ['concepto_saludo', 'concepto_pregunta_bienestar']
                        reasoning = "Patrón de saludo detectado."
                        logger.debug(f"[DEBUG TN SocialProtocol] Propuesta generada: {plan}")
                        return ThoughtProposal(self, plan, conf, reasoning)

            # Interacción social
            elif hypo_type == 'social_interaction':
                subtype = scenario.get('subtype')
                if subtype == 'informal_greeting':
                    conf = scenario.get('confidence', 0.8)
                    if self._concept_exists('concepto_saludo') and self._concept_exists('concepto_pregunta_bienestar'):
                        plan = ['concepto_saludo', 'concepto_pregunta_bienestar']
                        reasoning = "Saludo detectado."
                        return ThoughtProposal(self, plan, conf, reasoning)
                elif subtype == 'direct_wellbeing_question':
                    conf = scenario.get('confidence', 0.8)
                    plan = []
                    if self._concept_exists('concepto_respuesta_bienestar'):
                        plan.append('concepto_respuesta_bienestar')
                    if self._concept_exists('concepto_pregunta_reciproca'):
                        plan.append('concepto_pregunta_reciproca')
                    if plan:
                        reasoning = "Pregunta por bienestar."
                        return ThoughtProposal(self, plan, conf, reasoning)
                elif subtype == 'farewell':
                    conf = scenario.get('confidence', 0.8)
                    if self._concept_exists('concepto_despedida'):
                        plan = ['concepto_despedida']
                        reasoning = "Despedida detectada."
                        return ThoughtProposal(self, plan, conf, reasoning)

        logger.debug("[DEBUG TN SocialProtocol] No se generó ninguna propuesta")
        return None


class LogicalAnalystTN(BaseThinkingNeuron):
    """Punto de vista lógico: preguntas fácticas, operaciones matemáticas y edad."""
    
    def _concept_exists(self, concept_id: str) -> bool:
        """Verifica si un concepto existe en macro_state."""
        if self.engine and self.engine.macro_state:
            return concept_id in self.engine.macro_state.ids
        return False
    
    def _detect_language(self, ids: List[str]) -> str:
        """
        Detecta el idioma predominante entre los IDs dados (español o inglés).
        """
        spanish_indicators = ['mn_cero', 'mn_uno', 'mn_dos', 'mn_tres', 'mn_cuatro', 
                                'mn_cinco', 'mn_seis', 'mn_siete', 'mn_ocho', 'mn_nueve',
                                'mn_diez', 'mn_mas', 'mn_menos', 'mn_por', 'mn_dividido']
        
        english_indicators = ['mn_zero', 'mn_one', 'mn_two', 'mn_three', 'mn_four',
                               'mn_five', 'mn_six', 'mn_seven', 'mn_eight', 'mn_nine',
                               'mn_ten', 'mn_plus', 'mn_minus', 'mn_times', 'mn_divided_by']
        
        count_es = sum(1 for mid in ids if any(ind in mid for ind in spanish_indicators))
        count_en = sum(1 for mid in ids if any(ind in mid for ind in english_indicators))
        
        return 'es' if count_es >= count_en else 'en'

    def _extract_numbers_from_text(self, text: str) -> List[str]:
        """Intenta extraer IDs de números a partir de un string."""
        known_numbers = [
            'mn_cero', 'mn_uno', 'mn_dos', 'mn_tres', 'mn_cuatro', 'mn_cinco',
            'mn_seis', 'mn_siete', 'mn_ocho', 'mn_nueve', 'mn_diez',
            'mn_zero', 'mn_one', 'mn_two', 'mn_three', 'mn_four', 'mn_five',
            'mn_six', 'mn_seven', 'mn_eight', 'mn_nine', 'mn_ten'
        ]
        for num in known_numbers:
            if num in text:
                return [num]
        return []

    def propose(self, neural_state, retrieved_memories, context_hypotheses):
        logger.debug(f"[DEBUG TN LogicalAnalyst] Evaluando {len(context_hypotheses)} hipótesis")
        
        # ------------------------------------------------------------------
        # 0. AGE-RELATED QUESTION DETECTION
        # ------------------------------------------------------------------
        active_micros = neural_state.get('micro_neurons', {})
        age_words = ['mn_años', 'mn_edad', 'mn_age', 'mn_years', 'años', 'edad']
        if any(p in active_micros for p in age_words):
            if self._concept_exists('concepto_auto_revelacion_edad'):
                logger.debug(f"[DEBUG TN LogicalAnalyst] Detectada pregunta sobre edad")
                plan = ['concepto_auto_revelacion_edad']
                conf = 0.8
                return ThoughtProposal(self, plan, conf, "Pregunta sobre edad")

        # ------------------------------------------------------------------
        # 1. DIRECT DETECTION FROM MICRO-NEURONS
        # ------------------------------------------------------------------
        number_ids = [
            'mn_cero', 'mn_uno', 'mn_dos', 'mn_tres', 'mn_cuatro', 'mn_cinco',
            'mn_seis', 'mn_siete', 'mn_ocho', 'mn_nueve', 'mn_diez',
            'mn_zero', 'mn_one', 'mn_two', 'mn_three', 'mn_four', 'mn_five',
            'mn_six', 'mn_seven', 'mn_eight', 'mn_nine', 'mn_ten'
        ]
        operator_ids = [
            'mn_mas', 'mn_menos', 'mn_por', 'mn_dividido',
            'mn_plus', 'mn_minus', 'mn_times', 'mn_divided_by'
        ]
        
        active_numbers = [mid for mid in active_micros if active_micros[mid] and mid in number_ids]
        active_operators = [mid for mid in active_micros if active_micros[mid] and mid in operator_ids]
        
        logger.debug(f"[DEBUG TN LogicalAnalyst] Números detectados: {active_numbers}")
        logger.debug(f"[DEBUG TN LogicalAnalyst] Operadores detectados: {active_operators}")
        
        has_direct_evidence = (len(active_numbers) >= 2 and len(active_operators) >= 1)
        
        if has_direct_evidence:
            if self._concept_exists('concepto_resultado_calculo'):
                metadata = {
                    'numbers': active_numbers,
                    'operators': active_operators,
                    'language': self._detect_language(active_numbers + active_operators),
                    'origin': 'direct_input'
                }
                plan = ['concepto_resultado_calculo']
                conf = 0.95
                logger.debug(f"[DEBUG TN LogicalAnalyst] ⭐ Propuesta directa: {plan}")
                return ThoughtProposal(self, plan, conf, "Operación detectada directamente", metadata=metadata)

        # ------------------------------------------------------------------
        # 2. DETECTION FROM CALCULATION NEURONS
        # ------------------------------------------------------------------
        active_neurons = neural_state.get('neurons', {})
        for nid, is_active in active_neurons.items():
            if is_active and nid in ['n_pregunta_calculo_simple', 'n_pregunta_calculo_ingles']:
                if active_numbers:
                    logger.debug(f"[DEBUG TN LogicalAnalyst] Neurona {nid} activa con números")
                    if self._concept_exists('concepto_resultado_calculo'):
                        metadata = {
                            'numbers': active_numbers,
                            'operators': active_operators,
                            'language': self._detect_language(active_numbers + active_operators),
                            'origin': 'neuron_with_numbers'
                        }
                        return ThoughtProposal(self, ['concepto_resultado_calculo'], 0.85, "", metadata=metadata)
                else:
                    for h in context_hypotheses:
                        if h.get('type') == 'calculation_question':
                            key_elements = h.get('key_elements', [])
                            hyp_numbers = []
                            for elem in key_elements:
                                if elem in number_ids:
                                    hyp_numbers.append(elem)
                                elif any(c.isdigit() for c in elem):
                                    hyp_numbers.append(elem)
                            if len(hyp_numbers) >= 2:
                                logger.debug(f"[DEBUG TN LogicalAnalyst] Números en hipótesis: {hyp_numbers}")
                                if self._concept_exists('concepto_resultado_calculo'):
                                    metadata = {
                                        'numbers': hyp_numbers,
                                        'operators': ['mn_mas'],
                                        'language': 'es',
                                        'origin': 'hypothesis_calculation'
                                    }
                                    return ThoughtProposal(self, ['concepto_resultado_calculo'], 0.8, "", metadata=metadata)

        # ------------------------------------------------------------------
        # 3. PROCESS CONTEXT HYPOTHESES
        # ------------------------------------------------------------------
        for i, scenario in enumerate(context_hypotheses):
            hypo_type = scenario.get('type')
            
            if hypo_type == 'calculation_question' and not has_direct_evidence:
                key_elements = scenario.get('key_elements', [])
                hyp_numbers = [e for e in key_elements if e in number_ids or any(c.isdigit() for c in e)]
                if len(hyp_numbers) >= 2 and self._concept_exists('concepto_resultado_calculo'):
                    metadata = {
                        'numbers': hyp_numbers,
                        'operators': ['mn_mas'],
                        'language': 'es',
                        'origin': 'hypothesis_calculation'
                    }
                    return ThoughtProposal(self, ['concepto_resultado_calculo'], 0.8, "", metadata=metadata)

            if hypo_type == 'thinking_memory_focus':
                key_elements = scenario.get('key_elements', [])
                if key_elements and len(key_elements) == 1:
                    word_id = key_elements[0]
                    if word_id in ['mn_quien', 'mn_who']:
                        if self._concept_exists('concepto_auto_revelacion_identidad'):
                            plan = ['concepto_auto_revelacion_identidad']
                            conf = scenario.get('confidence', 0.8)
                            return ThoughtProposal(self, plan, conf, "")
                    elif word_id in ['mn_como', 'mn_how']:
                        if self._concept_exists('concepto_respuesta_bienestar'):
                            plan = ['concepto_respuesta_bienestar']
                            conf = scenario.get('confidence', 0.8)
                            return ThoughtProposal(self, plan, conf, "")
                    elif word_id in ['mn_que', 'mn_what']:
                        if self._concept_exists('concepto_clarificacion'):
                            plan = ['concepto_clarificacion']
                            conf = scenario.get('confidence', 0.8)
                            return ThoughtProposal(self, plan, conf, "")

            if hypo_type == 'factual_question':
                subtype = scenario.get('subtype')
                conf = scenario.get('confidence', 0.9)
                if subtype == 'identity_question':
                    if self._concept_exists('concepto_auto_revelacion_identidad'):
                        plan = ['concepto_auto_revelacion_identidad']
                        return ThoughtProposal(self, plan, conf, "")
                if subtype == 'capability_question':
                    if self._concept_exists('concepto_auto_revelacion_capacidad'):
                        plan = ['concepto_auto_revelacion_capacidad']
                        return ThoughtProposal(self, plan, conf, "")

        logger.debug("[DEBUG TN LogicalAnalyst] No se generó ninguna propuesta")
        return None


class AmbiguityDetectorTN(BaseThinkingNeuron):
    def _concept_exists(self, concept_id: str) -> bool:
        if self.engine and self.engine.macro_state:
            return concept_id in self.engine.macro_state.ids
        return False

    def propose(self, neural_state, retrieved_memories, context_hypotheses):
        num = len(context_hypotheses)
        logger.debug(f"[DEBUG TN AmbiguityDetector] Número de hipótesis: {num}")
        
        if num > 10:
            relevant_hypotheses = [h for h in context_hypotheses if h.get('type') != 'thinking_memory_focus']
            
            if len(relevant_hypotheses) >= 3:
                avg_conf = sum(h.get('confidence', 0) for h in relevant_hypotheses) / len(relevant_hypotheses)
                amb_confidence = min(0.85, avg_conf * (1 + 0.05 * len(relevant_hypotheses)))
                
                logger.debug(f"[DEBUG TN AmbiguityDetector] Hipótesis relevantes: {len(relevant_hypotheses)}, "
                      f"avg_conf={avg_conf:.2f}, amb_confidence={amb_confidence:.2f}")
                
                if amb_confidence > 0.75 and self._concept_exists('concepto_clarificacion'):
                    plan = ['concepto_clarificacion']
                    reasoning = f"{len(relevant_hypotheses)} interpretaciones posibles, ambigüedad {amb_confidence:.2f}."
                    return ThoughtProposal(self, plan, amb_confidence, reasoning)
        
        logger.debug("[DEBUG TN AmbiguityDetector] No se generó propuesta")
        return None


# --- 4. MacroTN (Master Synthesizer) ---
class MacroTN:
    def __init__(self, thinking_neurons: List[BaseThinkingNeuron]):
        self.thinking_neurons = thinking_neurons
        self.engine = None
        self.memory = None
        self.neural_system = None
        self.interconnectors = None

    def reasoning_cycle(self, neural_state: Dict, retrieved_memories: Dict,
                       context_hypotheses: List[Dict]) -> tuple:
        """
        Retorna (winning_plan, all_proposals, final_proposals)
        """
        logger.debug(f"[DEBUG MacroTN] Iniciando ciclo con {len(context_hypotheses)} hipótesis")
        
        # Recolectar propuestas
        proposals = []
        for tn in self.thinking_neurons:
            p = tn.propose(neural_state, retrieved_memories, context_hypotheses)
            if p:
                proposals.append(p)
                logger.debug(f"[DEBUG MacroTN] Propuesta de {p.source}: plan={p.conceptual_plan}, conf={p.confidence:.2f}")

        if not proposals:
            logger.debug("[DEBUG MacroTN] No hay propuestas, retornando None")
            return None, [], []

        # 1. Priorizar basado en macro‑neuronas activas
        active_macros = []
        if self.engine and self.engine.macro_state:
            macro_state = self.engine.macro_state
            for i, is_active in enumerate(macro_state.active):
                if is_active:
                    active_macros.append(macro_state.ids[i])
        logger.debug(f"[DEBUG MacroTN] Macro‑neuronas activas: {active_macros}")

        if active_macros:
            macro_proposals = []
            for p in proposals:
                if any(conc in active_macros for conc in p.conceptual_plan):
                    macro_proposals.append(p)
            if macro_proposals:
                best = max(macro_proposals, key=lambda x: x.confidence)
                logger.debug(f"[DEBUG MacroTN] Priorizando macro, mejor: {best.source}")
                return {"conceptual_plan": best.conceptual_plan, "metadata": best.metadata}, proposals, [best]

        # 2. Manejar ambigüedad
        amb_proposals = [p for p in proposals if p.source == 'AmbiguityDetectorTN']
        other_proposals = [p for p in proposals if p.source != 'AmbiguityDetectorTN']

        if other_proposals:
            best_other = max(other_proposals, key=lambda x: x.confidence)
            if best_other.confidence > 0.7:
                if amb_proposals:
                    amb_confidence = amb_proposals[0].confidence
                    if amb_confidence > 0.95:
                        logger.debug(f"[DEBUG MacroTN] Ambigüedad muy alta ({amb_confidence:.2f})")
                        return {"conceptual_plan": amb_proposals[0].conceptual_plan, "metadata": amb_proposals[0].metadata}, proposals, [amb_proposals[0]]
                logger.debug(f"[DEBUG MacroTN] Priorizando propuesta no ambigua: {best_other.source}")
                return {"conceptual_plan": best_other.conceptual_plan, "metadata": best_other.metadata}, proposals, [best_other]
            else:
                if amb_proposals:
                    amb_confidence = amb_proposals[0].confidence
                    if amb_confidence > best_other.confidence:
                        logger.debug(f"[DEBUG MacroTN] Ambigüedad ({amb_confidence:.2f}) > otras")
                        return {"conceptual_plan": amb_proposals[0].conceptual_plan, "metadata": amb_proposals[0].metadata}, proposals, [amb_proposals[0]]
                    else:
                        logger.debug(f"[DEBUG MacroTN] Mejor otra pese a baja confianza: {best_other.source}")
                        return {"conceptual_plan": best_other.conceptual_plan, "metadata": best_other.metadata}, proposals, [best_other]
                else:
                    logger.debug(f"[DEBUG MacroTN] Mayor confianza: {best_other.source}")
                    return {"conceptual_plan": best_other.conceptual_plan, "metadata": best_other.metadata}, proposals, [best_other]
        else:
            if amb_proposals:
                logger.debug("[DEBUG MacroTN] Solo ambigüedad disponible")
                return {"conceptual_plan": amb_proposals[0].conceptual_plan, "metadata": amb_proposals[0].metadata}, proposals, [amb_proposals[0]]
            else:
                best_proposal = max(proposals, key=lambda x: x.confidence)
                logger.debug(f"[DEBUG MacroTN] Mayor confianza: {best_proposal.source}")
                return {"conceptual_plan": best_proposal.conceptual_plan, "metadata": best_proposal.metadata}, proposals, [best_proposal]


# --- 5. Population Function ---
def populate_tns(memory=None, neural_system=None, interconnectors=None, engine=None):
    """
    Crea las Thinking Neurons y la MacroTN, inyectando dependencias.
    """
    tn_list = [
        SocialProtocolTN(),
        LogicalAnalystTN(),
        AmbiguityDetectorTN(),
    ]
    macro = MacroTN(tn_list)

    for tn in tn_list:
        tn.memory = memory
        tn.neural_system = neural_system
        tn.interconnectors = interconnectors
        tn.engine = engine

    macro.memory = memory
    macro.neural_system = neural_system
    macro.interconnectors = interconnectors
    macro.engine = engine

    logger.debug(f"[DEBUG] Pobladas {len(tn_list)} neuronas de pensamiento especializado.")
    return tn_list, macro