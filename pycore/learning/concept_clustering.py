# concept_clustering.py
import torch
import logging

from typing import List, Dict, Set, Tuple
from ..states.macro_state import MacroNeuronState
from ..states.micro_state import MicroNeuronState

class ConceptGenerator:
    """
    Módulo de aprendizaje no supervisado que crea nuevos conceptos (macro‑neuronas)
    basándose en la co‑activación de micro‑neuronas.
    """
    def __init__(self, micro_state: MicroNeuronState, macro_state: MacroNeuronState,
                 cooccurrence_threshold: float = 0.7, window: int = 10):
        self.micro_state = micro_state
        self.macro_state = macro_state
        self.threshold = cooccurrence_threshold
        self.window = window

        # Matriz de co‑ocurrencia (micro, micro)
        self.cooccurrences = torch.zeros(micro_state.n, micro_state.n, device=micro_state.device)

        # Historial de activaciones por contexto (últimas N frases)
        self.context_history = []  # lista de sets de IDs de micro activas
        
        # Configurar logger
        self.logger = logging.getLogger(__name__)

    def register_activations(self, active_micro_ids: List[str]):
        """
        Almacena las micro‑neuronas activas en este contexto.
        """
        indices = [self.micro_state.ids.index(mid) for mid in active_micro_ids]
        # Actualizar matriz de co‑ocurrencia
        for i in indices:
            for j in indices:
                if i != j:
                    self.cooccurrences[i, j] += 1.0
        # Guardar contexto
        self.context_history.append(set(active_micro_ids))
        if len(self.context_history) > self.window:
            self.context_history.pop(0)

    def detect_new_concepts(self) -> List[Tuple[Set[str], float]]:
        """
        Analiza la matriz de co‑ocurrencia para identificar grupos de palabras
        que suelen aparecer juntas y que aún no tienen un concepto asociado.
        Devuelve una lista de (set_of_ids, confidence).
        """
        # Normalizar co‑ocurrencias por frecuencias individuales
        individual_freqs = torch.diag(self.cooccurrences).clone()
        individual_freqs = torch.where(individual_freqs > 0, individual_freqs, torch.ones_like(individual_freqs))
        norm_matrix = self.cooccurrences / individual_freqs.unsqueeze(1)

        # Buscar grupos densos (método simple: umbral fijo)
        groups = []
        visited = set()
        for i in range(self.micro_state.n):
            if i in visited:
                continue
            # Buscar todas las micros que co‑ocurren con i por encima del umbral
            group = {i}
            for j in range(self.micro_state.n):
                if i != j and norm_matrix[i, j] > self.threshold:
                    group.add(j)
            if len(group) >= 2:  # al menos 2 palabras para formar concepto
                # Comprobar si este grupo ya tiene un concepto asignado
                group_ids = [self.micro_state.ids[idx] for idx in group]
                assigned_concepts = set()
                for mid in group_ids:
                    # Buscar en los metadatos de micro si tiene concepto
                    meta = self.micro_state.metadata[self.micro_state.ids.index(mid)]
                    if 'concept_id' in meta and meta['concept_id']:
                        assigned_concepts.add(meta['concept_id'])
                if not assigned_concepts:
                    confidence = norm_matrix[i, :][list(group)].mean().item()
                    groups.append((set(group_ids), confidence))
                visited.update(group)
        return groups
   
    def create_new_concept(self, word_ids: Set[str], confidence: float, engine) -> str:
        """
        Crea una nueva macro‑neurona (concepto) a partir de un conjunto de palabras.
        1. Crea una neurona intermedia que se activa con esas palabras.
        2. Crea la macro‑neurona que tiene como condición esa neurona.
        3. Asocia las palabras al concepto en sus metadatos.
        Devuelve el ID del nuevo concepto o None si falla.
        """
        # 1. Generar IDs únicos
        num_existing = len(engine.macro_state.ids)
        neuron_id = f"learned_neuron_{num_existing}"
        concept_id = f"learned_concept_{num_existing}"

        self.logger.debug(f"\n[CLUSTERING] Creando nuevo concepto: {concept_id}")
        self.logger.debug(f"             Palabras: {word_ids}")
        self.logger.debug(f"             Confianza: {confidence:.3f}")

        # 2. Crear la neurona intermedia
        # Convertir IDs de palabras a lista (para pasarlas como condiciones de la neurona)
        condition_micro_ids = list(word_ids)
        neuron_success = engine.add_neuron(
          new_id=neuron_id,
          new_name=f"Neuron for {concept_id}",
          condition_micro_ids=condition_micro_ids,
          exclusion_micro_ids=None,          # sin exclusiones inicialmente
          initial_threshold=confidence,            # usar la confianza como umbral
          metadata={'origin': 'clustering', 'target_concept': concept_id}
        )
        if not neuron_success:
          self.logger.error("[ERROR] No se pudo crear la neurona intermedia")
          return None

        # 3. Obtener el índice de la nueva neurona (está al final)
        neuron_idx = engine.neuron_state.N - 1

        # 4. Crear la macro‑neurona con esa neurona como condición
        # Las exclusiones pueden ser las mismas palabras (para inhibir si aparecen aisladas?).
        # Por simplicidad, dejamos exclusiones vacías.
        micro_indices = []
        for mid in word_ids:
          try:
            micro_indices.append(engine.micro_state.ids.index(mid))
          except ValueError:
            continue

        macro_success = engine.add_macro_neuron(
        	new_id=concept_id,
        	new_name=f"Learned concept {concept_id}",
        	condition_neuron_indices=[neuron_idx],
        	exclusion_micro_indices=micro_indices,   # opcional: las palabras inhiben el concepto
        	initial_threshold=confidence,
        	metadata={'origin': 'clustering', 'confidence': confidence}
    	  )
        if not macro_success:
          self.logger.error("[ERROR] No se pudo crear la macro‑neurona")
          return None

        # 5. Asociar cada palabra al nuevo concepto en sus metadatos
        for mid in word_ids:
          try:
              idx = engine.micro_state.ids.index(mid)
              engine.micro_state.metadata[idx]['concept_id'] = concept_id
          except ValueError:
              continue

        # 6. Opcional: añadir la relación en la memoria asociativa (grafo)
        for mid in word_ids:
          engine.memory.add_relationship(mid, concept_id, relationship_type=1.0)

        self.logger.debug(f"[CLUSTERING] Concepto {concept_id} creado exitosamente.")
        return concept_id

    def save_state(self, path):
      import pickle
      state = {
          'cooccurrences': self.cooccurrences.cpu().numpy(),
          'context_history': self.context_history,
          'threshold': self.threshold,
          'window': self.window
      }
      with open(path, 'wb') as f:
          pickle.dump(state, f)

    def load_state(self, path):
        import pickle
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.cooccurrences = torch.tensor(state['cooccurrences'], device=self.micro_state.device)
        self.context_history = state['context_history']
        self.threshold = state['threshold']
        self.window = state['window'] 