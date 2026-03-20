# grammar_adjudicator.py

from typing import List, Optional, Dict, Any

import torch
from .syntax_engine import SyntaxEngine
from .semantic_validator import SemanticValidator
# from .micro_neuron import MicroNeuronState
from ..states.macro_state import MacroNeuronState
# from .interconnector import InterconnectorState


class GrammarAdjudicator:
    """
    Gestiona las reglas de secuencia gramatical para la generación de respuestas.
    Ahora opera sobre conceptos (macro‑neuronas) usando una matriz de transiciones aprendida.
    """

    def __init__(self):
        self.syntax_engine = SyntaxEngine()
        self.semantic_validator = SemanticValidator()

    def get_valid_next_concepts(self, macro_state: MacroNeuronState,
                                current_concept_id: Optional[str] = None,
                                top_k: int = 5) -> List[str]:
        """
        Devuelve una lista de IDs de macro‑neuronas (conceptos) que pueden seguir al concepto actual,
        basándose en la matriz de transiciones aprendida (macro_state.transitions).
        Si current_concept_id es None, se devuelven conceptos que suelen iniciar frase (por ahora,
        simplemente los primeros top_k conceptos).
        """
        if macro_state is None:
            return []

        transitions = macro_state.transitions  # obtenemos la matriz del estado

        if current_concept_id is None:
            # Para inicio de frase, podríamos devolver los conceptos con mayor probabilidad de ser inicio.
            # Por simplicidad, devolvemos los primeros top_k conceptos.
            # En el futuro, podríamos tener un vector de "inicio" aprendido.
            return macro_state.ids[:top_k]

        # Buscar índice del concepto actual
        try:
            current_idx = macro_state.ids.index(current_concept_id)
        except ValueError:
            return []

        # Obtener las transiciones desde este concepto
        weights = transitions[current_idx]  # tensor de tamaño Nm

        # Seleccionar los top_k con mayor peso, ignorando pesos cero o negativos? (asumimos pesos no negativos)
        # Usamos torch.topk, pero si hay menos de top_k elementos con peso > 0, topk igual devolverá índices con peso bajo.
        # Podríamos filtrar después por peso > 0.
        k = min(top_k, len(weights))
        if k == 0:
            return []
        top_indices = torch.topk(weights, k).indices
        # Convertir a IDs, solo si el peso es positivo (opcional)
        result_ids = []
        for i in top_indices.tolist():
            if weights[i] > 0:  # opcional: ignorar transiciones con peso cero
                result_ids.append(macro_state.ids[i])
        return result_ids 