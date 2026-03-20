import collections
from typing import Dict, List, Any, Optional

# Asumimos que estas importaciones serán correctas en la nueva estructura
from ..states.micro_state import MicroNeuronState
from ..states.neuron_state import NeuronState
from ..states.macro_state import MacroNeuronState
from ..states.interconnector_state import InterconnectorState
from ..core.memory import Memory
from ..core.cognitive_engine import CognitiveEngine


class ContextSynthesizer:
    """
    Analiza las activaciones neuronales para inferir y construir dinámicamente
    un conjunto de hipótesis de contexto (escenarios). Adaptado para trabajar
    con los estados agrupados en tensores del nuevo sistema.
    """

    def __init__(self, engine: CognitiveEngine):
        """
        Inicializa el sintetizador con una referencia al motor cognitivo (engine),
        que contiene todos los estados (micro, neuronas, macro, interconectoras, memoria).
        """
        self.engine = engine

    def _extract_activations_dict(self, state, type: str) -> Dict[str, float]:
        """
        Convierte un estado de población (micro, neuronas, etc.) en un diccionario
        ID -> nivel de activación (float). Solo incluye aquellos con activación > 0.
        """
        if state is None:
            return {}
        activations = {}
        for i, id_ in enumerate(state.ids):
            level = state.activation_level[i].item()
            if level > 0:
                activations[id_] = level
        return activations

    def _infer_scenario_from_patterns(self, active_patterns: Dict[str, float],
                                      active_concepts: Dict[str, float]) -> Optional[Dict]:
        """
        Infiere un escenario a partir de patrones y conceptos activos.
        (Idéntica a la original, solo que recibe diccionarios ya convertidos.)
        """
        active_elements = []
        for pid, conf in active_patterns.items():
            active_elements.append({'id': pid, 'type': 'pattern', 'confidence': conf})
        for cid, conf in active_concepts.items():
            active_elements.append({'id': cid, 'type': 'concept', 'confidence': conf})

        if not active_elements:
            return None

        sorted_elements = sorted(active_elements, key=lambda x: x['confidence'], reverse=True)

        inferred_type = 'general'
        inferred_subtype = 'active_elements'
        inferred_confidence = sorted_elements[0]['confidence'] if sorted_elements else 0.0
        implications = ['requires_deep_analysis']

        # Ejemplo de inferencia social
        if sorted_elements and sorted_elements[0]['type'] == 'pattern' and sorted_elements[0]['confidence'] > 0.8:
            if 'saludo' in sorted_elements[0]['id'] or 'pregunta' in sorted_elements[0]['id']:
                inferred_type = 'social_interaction'
                inferred_subtype = sorted_elements[0]['id']
                implications.append('requires_contextual_response')

        return {
            'type': inferred_type,
            'subtype': inferred_subtype,
            'confidence': inferred_confidence,
            'active_elements': sorted_elements,
            'implications': implications
        }

    def synthesize(self,
                   neural_state: Optional[Dict] = None,
                   retrieved_memories: Optional[Dict] = None,
                   num_iterations: int = 5) -> List[Dict]:
        """
        Sintetiza hipótesis de contexto.
        Si no se proporciona neural_state, se obtiene del engine actual.
        retrieved_memories es un diccionario de memorias recuperadas (como antes).
        """
        if neural_state is None:
            # Construir a partir de los estados del engine
            micro_dict = self._extract_activations_dict(self.engine.micro_state, 'micro')
            neuron_dict = self._extract_activations_dict(self.engine.neuron_state, 'neuron')
            neural_state = {
                'micro_neurons': micro_dict,
                'neurons': neuron_dict
            }

        if retrieved_memories is None:
            retrieved_memories = {}

        # Obtener contenido de la memoria de pensamiento (se mantiene igual)
        thinking_memory_content = self.engine.memory.thinking.retrieve()

        # Generar hipótesis iniciales
        context_hypotheses = self._generate_initial_hypotheses(
            neural_state, retrieved_memories, thinking_memory_content)

        # Refinar iterativamente
        for i in range(num_iterations):
            context_hypotheses = self._refine_and_evaluate_hypotheses(
                context_hypotheses, neural_state, retrieved_memories, thinking_memory_content)

        return context_hypotheses

    def _generate_initial_hypotheses(self,
                                     neural_state: Dict,
                                     retrieved_memories: Dict,
                                     thinking_memory_content: List,
                                     activation_threshold: float = 0.6) -> List[Dict]:
        """
        Genera hipótesis iniciales a partir de los datos actuales.
        (Lógica similar a la original, pero accediendo a los objetos de estado a través del engine.)
        """
        hypotheses = []
        active_neurons = {nid: level for nid, level in neural_state.get('neurons', {}).items()
                          if level > activation_threshold}
        active_micro = {mnid: level for mnid, level in neural_state.get('micro_neurons', {}).items()
                        if level > activation_threshold}

        print(f"[DEBUG Sintetizador] Generando hipótesis con {len(active_neurons)} neuronas activas y {len(active_micro)} micros activas")

        # ------------------------------------------------------------------
        # 1. MacroNeuronas activas (a través del engine)
        # ------------------------------------------------------------------
        if self.engine.macro_state is not None:
            active_macros = []
            for i, is_active in enumerate(self.engine.macro_state.active):
                if is_active:
                    active_macros.append(self.engine.macro_state.ids[i])
            for mid in active_macros:
                hypo = {
                    'type': 'macro_neuron_focus',
                    'description': f'Contexto dominado por MacroNeurona: {mid}',
                    'confidence': 1.0,
                    'key_elements': [mid],
                    'evidence_support': {'macro': [mid]}
                }
                hypotheses.append(hypo)
                print(f"[DEBUG Sintetizador] Hipótesis añadida: type={hypo['type']}, elements={hypo['key_elements']}")

        # ------------------------------------------------------------------
        # 2. Patrones y hubs aprendidos
        # ------------------------------------------------------------------
        active_patterns = []
        active_hubs = []
        if self.engine.micro_state is not None:
            micro_ids = self.engine.micro_state.ids
            types = self.engine.micro_state.types
            for mnid, level in active_micro.items():
                try:
                    idx = micro_ids.index(mnid)
                except ValueError:
                    continue
                neuron_type = types[idx]
                if neuron_type == "pattern":
                    active_patterns.append(mnid)
                if "hub" in neuron_type:
                    active_hubs.append(mnid)

        for pid in active_patterns:
            hypo = {
                'type': 'learned_pattern',
                'description': f'Patrón aprendido detectado: {pid}',
                'confidence': active_micro.get(pid, 0.0),
                'key_elements': [pid],
                'evidence_support': {'pattern': [pid]},
                'explanation': ''
            }
            hypotheses.append(hypo)
            print(f"[DEBUG Sintetizador] Hipótesis añadida: type={hypo['type']}, elements={hypo['key_elements']}")

        for hid in active_hubs:
            hypo = {
                'type': 'active_hub',
                'description': f'Hub semántico activo: {hid}',
                'confidence': active_micro.get(hid, 0.0),
                'key_elements': [hid],
                'evidence_support': {'hub': [hid]},
                'explanation': ''
            }
            hypotheses.append(hypo)
            print(f"[DEBUG Sintetizador] Hipótesis añadida: type={hypo['type']}, elements={hypo['key_elements']}")

        # ------------------------------------------------------------------
        # 3. Hipótesis basadas en neuronas activas (UNA por todas las neuronas activas)
        # ------------------------------------------------------------------
        # ORIGINAL: Una sola hipótesis con la neurona principal
        if active_neurons:
            main_neuron = max(active_neurons.items(), key=lambda x: x[1])[0]
            hypo = {
                'type': 'neuron_focus',
                'description': f'Contexto centrado en la actividad de la neurona {main_neuron}.',
                'confidence': active_neurons[main_neuron],
                'key_elements': [main_neuron],
                'evidence_support': {'neurons': [main_neuron]}
            }
            hypotheses.append(hypo)
            print(f"[DEBUG Sintetizador] Hipótesis añadida: type={hypo['type']}, elements={hypo['key_elements']}")

            # MEJORA: Añadir también hipótesis específicas para neuronas de cálculo
            # (Esto es adicional, no reemplaza la original)
            calculation_neurons = [nid for nid in active_neurons
                                   if nid in ['n_pregunta_calculo_simple', 'n_pregunta_calculo_ingles']]
            for nid in calculation_neurons:
                calculation_hypo = {
                    'type': 'calculation_question',
                    'description': f'Detectada pregunta de cálculo: {nid}',
                    'confidence': active_neurons[nid],
                    'key_elements': [nid],
                    'evidence_support': {'calculation_neurons': [nid]}
                }
                hypotheses.append(calculation_hypo)
                print(f"[DEBUG Sintetizador] Hipótesis añadida (cálculo): type={calculation_hypo['type']}, elements={calculation_hypo['key_elements']}")

        # ------------------------------------------------------------------
        # 4. Hipótesis basadas en memorias recuperadas
        # ------------------------------------------------------------------
        if retrieved_memories:
            main_memory = max(retrieved_memories.items(),
                              key=lambda kv: kv[1].get('activation', 0.0))[0]
            hypo = {
                'type': 'associative_memory_focus',
                'description': f'Contexto influenciado por memoria asociativa clave: {main_memory}.',
                'confidence': retrieved_memories[main_memory].get('activation', 0.0),
                'key_elements': [main_memory],
                'evidence_support': {'memories': [main_memory]}
            }
            hypotheses.append(hypo)
            print(f"[DEBUG Sintetizador] Hipótesis añadida: type={hypo['type']}, elements={hypo['key_elements']}")

        # ------------------------------------------------------------------
        # 5. Hipótesis basadas en thinking memory (LIMITADAS A 5)
        # ------------------------------------------------------------------
        # ORIGINAL: Todas las que había en thinking memory
        # MEJORA: Solo las 5 más activas para no saturar

        # Ordenar thinking_memory_content por activación (de mayor a menor)
        sorted_items = sorted(thinking_memory_content,
                              key=lambda x: x.get('initial_activation', 0) if isinstance(x, dict) else 0,
                              reverse=True)

        # Tomar solo los 5 primeros
        for item in sorted_items[:5]:
            if isinstance(item, dict) and 'id' in item:
                hypo = {
                    'type': 'thinking_memory_focus',
                    'description': f'Contexto centrado en neurona clave de pensamiento: {item["id"]}.',
                    'confidence': item.get('initial_activation', 0.5),
                    'key_elements': [item['id']],
                    'evidence_support': {'thinking_neurons': [item['id']]}
                }
                hypotheses.append(hypo)
                print(f"[DEBUG Sintetizador] Hipótesis añadida: type={hypo['type']}, elements={hypo['key_elements']}")

        # ------------------------------------------------------------------
        # 6. Hipótesis combinadas (simplificadas) - se mantienen
        # ------------------------------------------------------------------
        if active_neurons and retrieved_memories:
            combined = []
            for nid in active_neurons:
                # Simulación: si la neurona tiene metadata con 'memory_concept_id'
                # (No implementado por simplicidad)
                pass
            if combined:
                hypotheses.append({
                    'type': 'combined_associative_focus',
                    'description': 'Contexto emergente de interacción neuronal y memoria.',
                    'confidence': 0.7,
                    'key_elements': list(set(combined)),
                    'evidence_support': {}
                })

        if active_neurons and thinking_memory_content:
            combined = []
            thinking_ids = [item.get('id') for item in thinking_memory_content if item.get('id')]
            for nid in active_neurons:
                if nid in thinking_ids:
                    combined.append(nid)
            if combined:
                hypotheses.append({
                    'type': 'combined_thinking_focus',
                    'description': 'Contexto emergente de interacción neuronal y thinking memory.',
                    'confidence': 0.8,
                    'key_elements': list(set(combined)),
                    'evidence_support': {}
                })

        # ------------------------------------------------------------------
        # 7. Eliminar duplicados simples
        # ------------------------------------------------------------------
        unique = []
        seen = set()
        for h in hypotheses:
            key = (h['type'], tuple(sorted(str(e) for e in h['key_elements'])))
            if key not in seen:
                unique.append(h)
                seen.add(key)

        print(f"[DEBUG Sintetizador] Total hipótesis generadas: {len(unique)}")
        return unique

    def _refine_and_evaluate_hypotheses(self,
                                        hypotheses: List[Dict],
                                        neural_state: Dict,
                                        retrieved_memories: Dict,
                                        thinking_memory_content: List,
                                        refinement_strength: float = 0.1,
                                        generation_threshold: float = 0.8) -> List[Dict]:
        """
        Refina hipótesis y genera nuevas. Versión simplificada que replica la original
        pero sin cálculos intensivos.
        """
        next_hypotheses = []
        active_neurons = neural_state.get('neurons', {})
        # active_micro no se usa en refinamiento por ahora

        for hyp in hypotheses:
            support = 0.0
            # Soporte de neuronas activas
            if 'neurons' in hyp.get('evidence_support', {}):
                for nid in hyp['evidence_support']['neurons']:
                    if nid in active_neurons and active_neurons[nid] > 0.5:
                        support += active_neurons[nid]
            if 'thinking_neurons' in hyp.get('evidence_support', {}):
                thinking_ids = [item.get('id') for item in thinking_memory_content if item.get('id')]
                for nid in hyp['evidence_support']['thinking_neurons']:
                    if nid in thinking_ids:
                        support += 0.8  # valor fijo

            # Soporte de memorias
            if 'memories' in hyp.get('evidence_support', {}):
                for mid in hyp['evidence_support']['memories']:
                    if mid in retrieved_memories:
                        support += retrieved_memories[mid].get('activation', 0.0)

            # Actualizar confianza (regla simple)
            support = min(1.0, support)
            hyp['confidence'] = max(0.0, min(1.0,
                                             hyp['confidence'] + (support - 0.5) * refinement_strength))
            if hyp['confidence'] > 0.1:
                next_hypotheses.append(hyp)

        # Generar nuevas hipótesis (placeholder)
        return next_hypotheses