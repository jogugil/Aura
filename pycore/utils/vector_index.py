import torch
from typing import List, Tuple, Dict, Any, Optional

########################################
## La clase VectorIndex usa búsqueda exhaustiva, lo cual es aceptable
##  para prototipos. 
## Para escalar, se podría reemplazar por FAISS más adelante.
#########################################

class VectorIndex:
    """
    Índice vectorial basado en PyTorch.
    Almacena embeddings y metadatos, permite búsqueda por similitud coseno.
    No usa KD-tree; en su lugar, realiza búsqueda exhaustiva (top‑k) sobre todos los vectores.
    Para escalar a millones, se puede sustituir por FAISS posteriormente.
    """
    def __init__(self, dim: int, device: torch.device = torch.device('cpu')):
        self.dim = dim
        self.device = device
        self.vectors = torch.empty(0, dim, device=device)   # (N, dim)
        self.ids: List[Any] = []
        self.metadata: List[Dict[str, Any]] = []
        self.id_to_index: Dict[Any, int] = {}
        self._removed_ids: set = set()

    def add_vector(self, vector_id: Any, vector: List[float], metadata: Optional[Dict] = None):
        """Añade un vector al índice."""
        if metadata is None:
            metadata = {}
        vec = torch.tensor(vector, dtype=torch.float32, device=self.device)
        if vec.shape[0] != self.dim:
            raise ValueError(f"Vector dimension mismatch: expected {self.dim}, got {vec.shape[0]}")
        vec = vec.unsqueeze(0)  # (1, dim)
        self.vectors = torch.cat([self.vectors, vec], dim=0)
        self.ids.append(vector_id)
        self.metadata.append(metadata)
        self.id_to_index[vector_id] = len(self.ids) - 1
        if vector_id in self._removed_ids:
            self._removed_ids.remove(vector_id)

    def get_vector(self, vector_id: Any) -> Optional[torch.Tensor]:
        """Devuelve el vector por ID."""
        if vector_id not in self.id_to_index:
            return None
        idx = self.id_to_index[vector_id]
        return self.vectors[idx].clone()

    def get_metadata(self, vector_id: Any) -> Optional[Dict]:
        if vector_id not in self.id_to_index:
            return None
        idx = self.id_to_index[vector_id]
        return self.metadata[idx].copy()

    def search_similar(self, query_vector: List[float], top_k: int = 5,
                       grammar_category: Optional[str] = None) -> List[Tuple[Any, float]]:
        """
        Busca los top_k vectores más similares por coseno.
        Si grammar_category no es None, filtra por metadatos['grammar_category'].
        """
        if self.vectors.shape[0] == 0:
            return []
        query = torch.tensor(query_vector, dtype=torch.float32, device=self.device)
        query = query.unsqueeze(0)  # (1, dim)

        # Normalizar vectores para similitud coseno
        vectors_norm = self.vectors / (torch.norm(self.vectors, dim=1, keepdim=True) + 1e-8)
        query_norm = query / (torch.norm(query) + 1e-8)

        sim = torch.mm(query_norm, vectors_norm.t()).squeeze(0)  # (N,)

        # Obtener top_k índices
        top_sim, top_indices = torch.topk(sim, min(top_k, len(sim)))

        results = []
        for idx, score in zip(top_indices.cpu().numpy(), top_sim.cpu().numpy()):
            if idx >= len(self.ids):
                continue
            vid = self.ids[idx]
            if vid in self._removed_ids:
                continue
            meta = self.metadata[idx]
            if grammar_category is not None and meta.get('grammar_category') != grammar_category:
                continue
            results.append((vid, float(score)))
        return results

    def remove_vector(self, vector_id: Any):
        """Marca un vector como eliminado (no se devuelve en búsquedas)."""
        if vector_id in self.id_to_index:
            self._removed_ids.add(vector_id)

    def __len__(self):
        return len(self.ids) - len(self._removed_ids)

    def size(self):
        return len(self)


# Para compatibilidad con el código original
embedding_index = VectorIndex(dim=64)

class IndexManager:
    def optimize_all(self):
        pass
    def get_all_stats(self):
        return {}

index_manager = IndexManager() 