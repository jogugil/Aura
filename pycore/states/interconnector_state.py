import torch
from typing import List, Dict, Any, Optional, Tuple

class InterconnectorState:
    """
    Contiene todas las neuronas interconectoras en forma de tensores.
    - K: número de interconectoras
    - C: número total de conceptos (todas las entidades con ID: micro, neuronas, macro, etc.)
    """
    def __init__(self,
                 ids: List[str],
                 connected_concept_ids: List[List[str]],   # lista de listas de IDs de conceptos a los que conecta
                 global_concept_to_idx: Dict[str, int],          # mapeo global ID → índice
                 embeddings: Optional[torch.Tensor] = None,      # (K, dim) opcional
                 rules: Optional[List[Dict[str, Any]]] = None,  # metadatos por interconectora
                 embedding_dim: int = 64,
                 device: torch.device = torch.device('cpu'),
                 dtype: torch.dtype = torch.float32):
        """
        Args:
            ids: lista de identificadores de las interconectoras
            connected_concept_ids: para cada interconectora, lista de IDs de conceptos que conecta
            global_concept_to_idx: diccionario que mapea cualquier ID de concepto a su índice global
            embeddings: si se proporciona, tensor (K, embedding_dim); si no, se generan aleatoriamente
            rules: lista de diccionarios con reglas/metadatos por interconectora
            embedding_dim: dimensión de los embeddings (por defecto 64)
        """
        self.device = device
        self.dtype = dtype
        self.dim = embedding_dim
        self.K = len(ids)
        self.C = len(global_concept_to_idx)  # número total de conceptos

        # Metadatos (no tensoriales)
        self.ids = ids
        if rules is None:
            rules = [{} for _ in range(self.K)]
        self.rules = rules

        # Embeddings
        if embeddings is not None:
            self.embeddings = embeddings.to(device, dtype)
            assert self.embeddings.shape == (self.K, self.dim)
        else:
            # Generar embeddings aleatorios uniformes en [-1, 1]
            self.embeddings = torch.empty(self.K, self.dim, device=device, dtype=dtype).uniform_(-1, 1)

        # Construir matriz de conexiones dispersa (K, C) con 1 en las posiciones conectadas
        rows = []
        cols = []
        for i, concept_list in enumerate(connected_concept_ids):
            for concept_id in concept_list:
                idx = global_concept_to_idx.get(concept_id)
                if idx is not None:
                    rows.append(i)
                    cols.append(idx)
        if rows:
            indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
            values = torch.ones(len(rows), device=device, dtype=dtype)
            self.connections = torch.sparse_coo_tensor(indices, values, (self.K, self.C), device=device)
        else:
            self.connections = torch.sparse_coo_tensor(torch.zeros(2, 0, dtype=torch.long), torch.empty(0), (self.K, self.C), device=device)

    def to(self, device: torch.device):
        """Mueve todos los tensores a otro dispositivo."""
        self.device = device
        self.embeddings = self.embeddings.to(device)
        self.connections = self.connections.to(device)
        return self

    # ------------------------------------------------------------------
    # Funciones de utilidad
    # ------------------------------------------------------------------
    def get_relevant(self, active_concept_indices: torch.Tensor) -> torch.Tensor:
        """
        Dado un tensor 1D con índices de conceptos activos, devuelve un tensor booleano
        de tamaño K indicando qué interconectoras son relevantes (conectan con al menos uno).
        """
        if self.connections._nnz() == 0:
            return torch.zeros(self.K, dtype=torch.bool, device=self.device)

        # Crear vector indicador de conceptos activos (1 donde activos)
        concepts_vec = torch.zeros(self.C, device=self.device, dtype=self.dtype)
        if len(active_concept_indices) > 0:
            concepts_vec.scatter_(0, active_concept_indices, 1.0)

        # Producto matriz‑vector: (K, C) * (C,) -> (K,) número de coincidencias por interconectora
        matches = torch.sparse.mm(self.connections, concepts_vec.unsqueeze(1)).squeeze(1)  # (K,)
        return matches > 0

    def similarity_with_embedding(self, other_embedding: torch.Tensor) -> torch.Tensor:
        """
        Calcula la similitud coseno entre el embedding de cada interconectora y otro embedding dado.
        other_embedding: tensor 1D de dimensión dim
        Retorna: tensor (K,) con las similitudes.
        """
        # Normalizar embeddings
        emb_norm = self.embeddings / (torch.norm(self.embeddings, dim=1, keepdim=True) + 1e-8)
        other_norm = other_embedding / (torch.norm(other_embedding) + 1e-8)
        return torch.mv(emb_norm, other_norm)   # (K,)

    def similarity_between_interconnectors(self, idx1: int, idx2: int) -> float:
        """Similitud coseno entre dos interconectoras por índice."""
        emb1 = self.embeddings[idx1]
        emb2 = self.embeddings[idx2]
        return torch.dot(emb1, emb2) / (torch.norm(emb1) * torch.norm(emb2) + 1e-8)

    def has_connection(self, concept1_id: str, concept2_id: str, global_concept_to_idx: Dict[str, int]) -> bool:
        """
        Verifica si existe alguna interconectora que conecte ambos conceptos.
        Útil para ajustes de confianza.
        """
        idx1 = global_concept_to_idx.get(concept1_id)
        idx2 = global_concept_to_idx.get(concept2_id)
        if idx1 is None or idx2 is None:
            return False
        # Buscar en la matriz de conexiones si hay alguna fila con 1 en ambas columnas
        # Esto es más eficiente si convertimos a denso para la consulta, pero podemos usar la lógica de índices.
        # Para simplificar, usamos el método de producto: (connections * e1) * e2 > 0
        e1 = torch.zeros(self.C, device=self.device, dtype=self.dtype)
        e2 = torch.zeros(self.C, device=self.device, dtype=self.dtype)
        e1[idx1] = 1.0
        e2[idx2] = 1.0
        # Vector de activación de interconectoras que conectan con concepto1
        act1 = torch.sparse.mm(self.connections, e1.unsqueeze(1)).squeeze(1)  # (K,)
        act2 = torch.sparse.mm(self.connections, e2.unsqueeze(1)).squeeze(1)  # (K,)
        return (act1 * act2).sum() > 0


def create_interconnector_state_from_ids(
    inter_ids: List[str],
    connected_concept_ids: List[List[str]],
    global_concept_to_idx: Dict[str, int],
    rules: Optional[List[Dict]] = None,
    embedding_dim: int = 64,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32
) -> InterconnectorState:
    """
    Crea el estado de interconectoras a partir de listas de IDs.
    """
    if rules is None:
        rules = [{} for _ in inter_ids]
    else:
        assert len(rules) == len(inter_ids), "rules debe tener la misma longitud que inter_ids"

    return InterconnectorState(
        ids=inter_ids,
        connected_concept_ids=connected_concept_ids,
        global_concept_to_idx=global_concept_to_idx,
        rules=rules,
        embedding_dim=embedding_dim,
        device=device,
        dtype=dtype
    ) 