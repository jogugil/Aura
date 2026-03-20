# transition_learning.py
import torch
import logging
from typing import List

logger = logging.getLogger(__name__)

def reinforce_transitions_(self, concept_sequence_ids: List[str], success: bool, rate: float = 0.05):
        """
        Refuerza o debilita las transiciones entre conceptos consecutivos en una secuencia.
        
        Args:
            concept_sequence_ids: lista de IDs de macro‑neuronas en el orden en que aparecieron
            success: True si la respuesta fue exitosa, False si no
            rate: magnitud del cambio (positiva para éxito, negativa para fracaso)
        """
        logger.info(f"Reforzando transiciones para secuencia de {len(concept_sequence_ids)} conceptos, éxito={success}")
        
        if self.macro_state is None or self.transitions is None:
            logger.error("No se puede reforzar: macro_state o transitions no inicializado")
            return
            
        if len(concept_sequence_ids) < 2:
            logger.debug("Secuencia demasiado corta para reforzar transiciones")
            return

        # Convertir IDs a índices
        idxs = []
        for cid in concept_sequence_ids:
            if cid in self.macro_state.ids:
                idxs.append(self.macro_state.ids.index(cid))
                logger.debug(f"  Concepto {cid} encontrado en índice {idxs[-1]}")
            else:
                logger.debug(f"  Concepto {cid} no encontrado en macro_state, cancelando")
                return

        adjustment = rate if success else -rate
        logger.debug(f"Ajuste a aplicar: {adjustment}")

        transitions_updated = 0
        for i in range(len(idxs)-1):
            source = idxs[i]
            target = idxs[i+1]
            before = self.transitions[source, target].item()
            self.transitions[source, target] += adjustment
            # Mantener dentro de rango [0, 1]
            self.transitions[source, target] = torch.clamp(self.transitions[source, target], 0.0, 1.0)
            after = self.transitions[source, target].item()
            logger.debug(f"  Transición {source}->{target}: {before:.3f} -> {after:.3f}")
            transitions_updated += 1

        logger.debug(f"Transiciones actualizadas: {transitions_updated}")

def reinforce_transitions(engine, concept_sequence_ids: List[str], success: bool, rate: float = 0.05):
    """
    Refuerza o debilita las transiciones entre conceptos consecutivos.
    """
    if engine.macro_state is None or engine.transitions is None:
        logger.error("No se puede reforzar: macro_state o transitions no inicializado")
        return

    if len(concept_sequence_ids) < 2:
        return

    idxs = []
    for cid in concept_sequence_ids:
        if cid in engine.macro_state.ids:
            idxs.append(engine.macro_state.ids.index(cid))
        else:
            return

    adjustment = rate if success else -rate
    for i in range(len(idxs)-1):
        source = idxs[i]
        target = idxs[i+1]
        engine.transitions[source, target] += adjustment
        engine.transitions[source, target] = torch.clamp(engine.transitions[source, target], 0.0, 1.0)