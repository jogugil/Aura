# Clase base para la memoria de la IA
from __future__ import annotations
import torch
import logging

from typing import Dict, List, Any, Optional

# Configurar logger
logger = logging.getLogger(__name__)

def register_in_memory(memory: Memory, data: Any, level: str = "thinking"):
    """
    Función de compatibilidad para registrar en memoria.
    """
    memory.add_to_memory(data, level)

# ----------------------------------------------------------------------
# Memorias simples (no vectorizadas, se mantienen como Python lists)
# ----------------------------------------------------------------------
class ShortTermMemory:
    """Memoria a corto plazo con capacidad fija."""
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.memory = []

    def add(self, item: Any):
        """Añade un item a la memoria a corto plazo."""
        self.memory.append(item)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)

    def retrieve(self, query: Any = None) -> List[Any]:
        """Recupera items de la memoria a corto plazo."""
        return self.memory

    def clear(self):
        """Limpia la memoria a corto plazo."""
        self.memory = []


class MediumTermMemory:
    """Memoria a medio plazo (acceso por clave)."""
    def __init__(self):
        self.memory = {}

    def add(self, key: str, item: Any):
        """Añade un item con una clave a la memoria a medio plazo."""
        self.memory[key] = item

    def retrieve(self, key: str) -> Optional[Any]:
        """Recupera un item por su clave de la memoria a medio plazo."""
        return self.memory.get(key)

    def clear(self):
        """Limpia la memoria a medio plazo."""
        self.memory = {}


class LongTermMemory:
    """Memoria a largo plazo (simple lista)."""
    def __init__(self):
        self.memory = []

    def add(self, item: Any):
        """Añade un item a la memoria a largo plazo."""
        self.memory.append(item)

    def retrieve(self, query: Any = None) -> List[Any]:
        """Recupera items de la memoria a largo plazo."""
        return self.memory

    def clear(self):
        """Limpia la memoria a largo plazo."""
        self.memory = []


class ThinkingMemory:
    """Memoria de pensamiento (temporal, para el ciclo actual)."""
    def __init__(self):
        self.memory = []

    def add(self, item: Any):
        """Añade un item a la memoria de pensamiento."""
        self.memory.append(item)

    def retrieve(self, query: Any = None) -> List[Any]:
        """Recupera items de la memoria de pensamiento."""
        return self.memory

    def clear(self):
        """Limpia la memoria de pensamiento."""
        self.memory = []


# ----------------------------------------------------------------------
# Memoria principal con grafo conceptual vectorizado
# ----------------------------------------------------------------------
class Memory:
    """
    Memoria jerárquica con grafo conceptual para recuperación asociativa.
    El grafo se almacena como matriz de adyacencia dispersa sobre índices de conceptos.
    """
    def __init__(self, device: torch.device = torch.device('cpu')):
        # Memorias jerárquicas
        self.short_term = ShortTermMemory()
        self.medium_term = MediumTermMemory()
        self.long_term = LongTermMemory()
        self.thinking = ThinkingMemory()

        # Mapeo global de conceptos (ID -> índice)
        self.concept_to_idx: Dict[str, int] = {}
        self.idx_to_concept: Dict[int, str] = {}

        # Metadatos de conceptos (listas paralelas a los índices)
        self.concept_metadata: List[Dict[str, Any]] = []  # metadatos en orden de índice

        # Matriz de adyacencia (C, C) dispersa, con valores = tipo de relación (por ahora 1)
        self.adjacency: Optional[torch.Tensor] = None  # matriz dispersa COO
        self.device = device

    def _get_or_create_index(self, concept_id: str) -> int:
        """Obtiene el índice de un concepto; si no existe, lo crea."""
        if concept_id in self.concept_to_idx:
            return self.concept_to_idx[concept_id]
        idx = len(self.concept_to_idx)
        self.concept_to_idx[concept_id] = idx
        self.idx_to_concept[idx] = concept_id
        self.concept_metadata.append({})
        # Expandir matriz de adyacencia (se reconstruirá al añadir relaciones)
        # Por simplicidad, no la expandimos ahora; la reconstruiremos al añadir relaciones
        return idx

    def add_concept(self, concept_id: str, metadata: Optional[Dict] = None):
        """Añade un nodo concepto al grafo."""
        idx = self._get_or_create_index(concept_id)
        if metadata:
            self.concept_metadata[idx] = metadata
        # La matriz de adyacencia se actualizará cuando se añadan relaciones

    def add_relationship(self, concept1_id: str, concept2_id: str,
                         relationship_type: Any = 1, metadata: Optional[Dict] = None):
        """
        Añade una relación dirigida concept1_id -> concept2_id.
        relationship_type se almacena como peso en la matriz (por ahora 1).
        """
        i = self._get_or_create_index(concept1_id)
        j = self._get_or_create_index(concept2_id)

        # Construir lista de aristas
        # Almacenamos en un diccionario temporal para luego crear la matriz dispersa
        if not hasattr(self, '_edges'):
            self._edges = []  # lista de (i, j, weight)
        weight = 1.0 if relationship_type is True else float(relationship_type)
        self._edges.append((i, j, weight))

        # Reconstruir matriz de adyacencia
        self._rebuild_adjacency()

    def _rebuild_adjacency(self):
        """Reconstruye la matriz de adyacencia dispersa a partir de _edges."""
        if not hasattr(self, '_edges') or not self._edges:
            self.adjacency = None
            return
        C = len(self.concept_to_idx)
        edges = torch.tensor(self._edges, dtype=torch.long, device=self.device)
        indices = edges[:, :2].T  # (2, E)
        values = edges[:, 2].to(self.device)
        self.adjacency = torch.sparse_coo_tensor(indices, values, (C, C), device=self.device)

    def retrieve_associative(self,
                             query_concepts: List[str],
                             activation_levels: List[float],
                             depth_limit: int = 3,
                             activation_threshold: float = 0.3,
                             propagation_factor: float = 0.7) -> Dict[str, Dict]:
        """
        Recupera conceptos relacionados mediante propagación en el grafo.
        Versión vectorizada: usa multiplicación de matrices dispersas para simular la propagación.

        Args:
            query_concepts: lista de IDs de conceptos iniciales
            activation_levels: lista de niveles de activación inicial (misma longitud)
            depth_limit: número máximo de pasos de propagación
            activation_threshold: umbral mínimo de activación para considerar
            propagation_factor: factor de decaimiento por paso

        Returns:
            dict: {concept_id: {'metadata': ..., 'activation': ..., 'depth': ...}}
        """
        if self.adjacency is None or len(query_concepts) == 0:
            return {}

        # Convertir query_concepts a índices
        idx_list = []
        acts = []
        for cid, act in zip(query_concepts, activation_levels):
            if cid in self.concept_to_idx and act > activation_threshold:
                idx_list.append(self.concept_to_idx[cid])
                acts.append(act)

        if not idx_list:
            return {}

        C = len(self.concept_to_idx)
        device = self.device

        # Vector de activación inicial (C,)
        activation = torch.zeros(C, device=device, dtype=torch.float32)
        indices = torch.tensor(idx_list, device=device, dtype=torch.long)
        values = torch.tensor(acts, device=device, dtype=torch.float32)
        activation.scatter_(0, indices, values)

        # Para almacenar la máxima activación por nodo (incluyendo profundidad)
        max_activation = activation.clone()
        depth = torch.zeros(C, device=device, dtype=torch.long)
        depth[indices] = 0

        # Propagación por niveles
        for d in range(1, depth_limit + 1):
            # Multiplicar matriz de adyacencia por el vector de activación actual
            # activation_actual = activation de los nodos que alcanzaron profundidad d-1
            # Queremos propagar solo desde los que se activaron en el paso anterior.
            # Para simplificar, propagamos desde todos, pero decayendo.
            new_activation = torch.sparse.mm(self.adjacency, activation.unsqueeze(1)).squeeze(1) * propagation_factor
            new_activation = torch.where(new_activation > activation_threshold, new_activation, torch.zeros_like(new_activation))

            # Actualizar máximos
            mask = new_activation > max_activation
            max_activation = torch.where(mask, new_activation, max_activation)
            depth = torch.where(mask, torch.full_like(depth, d), depth)

            activation = new_activation
            if activation.sum() == 0:
                break

        # Convertir resultados a diccionario
        results = {}
        active_indices = torch.where(max_activation > activation_threshold)[0].cpu().numpy()
        for idx in active_indices:
            cid = self.idx_to_concept[idx]
            results[cid] = {
                'metadata': self.concept_metadata[idx],
                'activation': max_activation[idx].item(),
                'depth': depth[idx].item()
            }
        return results

    # ------------------------------------------------------------------
    # Métodos de interfaz con las memorias jerárquicas
    # ------------------------------------------------------------------
    def add_to_memory(self, item: Any, memory_level: str = "short"):
        """Añade un item al nivel de memoria especificado."""
        if memory_level == "short":
            self.short_term.add(item)
        elif memory_level == "medium":
            key = str(item)  # simplificado
            self.medium_term.add(key, item)
        elif memory_level == "long":
            self.long_term.add(item)
        elif memory_level == "thinking":
            self.thinking.add(item)
        else:
            logger.warning(f"Nivel de memoria desconocido: '{memory_level}'")
    
    def retrieve_from_memory(self, query: Any, memory_level: str = "short") -> Any:
        """Recupera items del nivel de memoria especificado."""
        if memory_level == "short":
            return self.short_term.retrieve(query)
        elif memory_level == "medium":
            return self.medium_term.retrieve(str(query))
        elif memory_level == "long":
            return self.long_term.retrieve(query)
        elif memory_level == "thinking":
            return self.thinking.retrieve(query)
        else:
            logger.warning(f"Nivel de memoria desconocido: '{memory_level}'")
            return []

    def clear_memory(self, memory_level: str = "all"):
        """Limpia el nivel de memoria especificado (o todos si 'all')."""
        if memory_level in ("short", "all"):
            self.short_term.clear()
        if memory_level in ("medium", "all"):
            self.medium_term.clear()
        if memory_level in ("long", "all"):
            self.long_term.clear()
        if memory_level in ("thinking", "all"):
            self.thinking.clear()

    # ------------------------------------------------------------------
    # Métodos para compatibilidad con código existente
    # ------------------------------------------------------------------
    def add_concept_original(self, concept_id, metadata=None):
        """Wrapper para mantener nombre original."""
        self.add_concept(concept_id, metadata)

    def add_relationship_original(self, concept1_id, concept2_id, relationship_type, metadata=None):
        self.add_relationship(concept1_id, concept2_id, relationship_type, metadata) 
    
    def register_in_memory(self, data, memory_level="short"):
        """Registra un dato en el nivel de memoria especificado."""
        self.add_to_memory(data, memory_level)

    def search_in_memory(self, query, memory_level="short"):
        """Busca unself, dato en el nivel de memoria especificado."""
        return self.retrieve_from_memory(query, memory_level)
