import torch
import logging
from typing import List, Dict, Any, Optional, Tuple

# Configurar logger
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Estado agrupado de todas las neuronas
# ----------------------------------------------------------------------
class NeuronState:
    """
    Contiene todas las neuronas en forma de tensores.
    - N: número de neuronas
    - M: número total de micro‑neuronas (dimensión de las condiciones)
    """
    def __init__(self,
                 ids: List[str],
                 names: List[str],
                 condition_indices: List[List[int]],   # lista de listas de índices de micro‑neuronas
                 exclusion_indices: List[List[int]],   # lista de listas de índices de exclusión
                 num_micro: int,                          # número total de micro‑neuronas
                 initial_weights: Optional[torch.Tensor] = None,
                 threshold: float = 0.6,
                 decay_rate: float = 0.25,
                 device: torch.device = torch.device('cpu'),
                 dtype: torch.dtype = torch.float32):
        """
        Args:
            ids: lista de identificadores (strings)
            names: lista de nombres descriptivos (strings)
            condition_indices: para cada neurona, lista de índices de micro‑neuronas que activan
            exclusion_indices: para cada neurona, lista de índices de micro‑neuronas que excluyen
            num_micro: número total de micro‑neuronas en el sistema
            initial_weights: si se proporciona, debe ser un tensor (N, num_micro) con los pesos
            threshold: valor por defecto para el umbral de activación
            decay_rate: valor por defecto para la tasa de decaimiento
        """
        self.device = device
        self.dtype = dtype
        self.N = len(ids)
        self.M = num_micro   # usar el número total de micro‑neuronas

        # Metadatos (no tensoriales)
        self.ids = ids
        self.names = names
        self.metadata = [{} for _ in range(self.N)]

        # Matriz de pesos (dispersa COO) de tamaño (N, M)
        if initial_weights is not None:
            self.weights = initial_weights.to(device, dtype)
        else:
            # Construir matriz dispersa a partir de condition_indices
            rows = []
            cols = []
            values = []
            for i, conds in enumerate(condition_indices):
                for j in conds:
                    rows.append(i)
                    cols.append(j)
                    weight = torch.empty(1).uniform_(0.3, 1.0).item()
                    values.append(weight)
            if rows:
                indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
                values = torch.tensor(values, dtype=dtype, device=device)
                self.weights = torch.sparse_coo_tensor(indices, values, (self.N, self.M), device=device, dtype=dtype)
            else:
                self.weights = torch.sparse_coo_tensor(torch.zeros(2,0, dtype=torch.long), torch.empty(0), (self.N, self.M), device=device)

        # Matriz de exclusiones (dispersa booleana) de tamaño (N, M)
        excl_rows = []
        excl_cols = []
        for i, exc in enumerate(exclusion_indices):
            for j in exc:
                excl_rows.append(i)
                excl_cols.append(j)
        if excl_rows:
            excl_indices = torch.tensor([excl_rows, excl_cols], dtype=torch.long, device=device)
            excl_values = torch.ones(len(excl_rows), dtype=dtype, device=device)
            self.exclusions = torch.sparse_coo_tensor(excl_indices, excl_values, (self.N, self.M), device=device, dtype=dtype)
        else:
            self.exclusions = torch.sparse_coo_tensor(torch.zeros(2,0, dtype=torch.long), torch.empty(0), (self.N, self.M), device=device)

        # Parámetros escalares por neurona
        if isinstance(threshold, float):
            self.activation_threshold = torch.full((self.N,), threshold, device=device, dtype=dtype)
        else:
            self.activation_threshold = threshold.to(device, dtype)

        if isinstance(decay_rate, float):
            self.decay_rate = torch.full((self.N,), decay_rate, device=device, dtype=dtype)
        else:
            self.decay_rate = decay_rate.to(device, dtype)

        # Estado variable
        self.activation_level = torch.zeros(self.N, device=device, dtype=dtype)
        self.active = torch.zeros(self.N, dtype=torch.bool, device=device)

        # Para atención contextual (frecuencia de activación de micro‑neuronas)
        self.micro_frequency = torch.zeros(self.M, device=device, dtype=dtype)

        # Historial (no tensorial, opcional)
        self.activation_history = [[] for _ in range(self.N)]
        # Guardar condition_indices para inhibición lateral
        self.condition_indices = condition_indices
        # Configurar logger
        self.logger = logging.getLogger(__name__)

    def to(self, device: torch.device):
        """Mueve todos los tensores a otro dispositivo."""
        self.device = device
        self.weights = self.weights.to(device)
        self.exclusions = self.exclusions.to(device)
        self.activation_threshold = self.activation_threshold.to(device)
        self.decay_rate = self.decay_rate.to(device)
        self.activation_level = self.activation_level.to(device)
        self.active = self.active.to(device)
        self.micro_frequency = self.micro_frequency.to(device)
        return self

# ----------------------------------------------------------------------
# Función de ayuda para crear estado a partir de listas de condiciones (con IDs)
# ----------------------------------------------------------------------
def create_neuron_state_from_ids(
    neuron_ids: List[str],
    names: List[str],
    condition_ids: List[List[str]],      # lista de listas de IDs de micro‑neuronas
    exclusion_ids: List[List[str]],      # lista de listas de IDs de micro‑neuronas
    micro_id_to_index: Dict[str, int],     # mapeo de ID de micro‑neurona a índice
    num_micro: int,                         # número total de micro‑neuronas
    threshold: float = 0.6,
    decay_rate: float = 0.25,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32
) -> NeuronState:
    condition_indices = []
    for conds in condition_ids:
        indices = [micro_id_to_index[cid] for cid in conds if cid in micro_id_to_index]
        condition_indices.append(indices)

    exclusion_indices = []
    for exc in exclusion_ids:
        indices = [micro_id_to_index[eid] for eid in exc if eid in micro_id_to_index]
        exclusion_indices.append(indices)

    return NeuronState(
        ids=neuron_ids,
        names=names,
        condition_indices=condition_indices,
        exclusion_indices=exclusion_indices,
        num_micro=num_micro,
        threshold=threshold,
        decay_rate=decay_rate,
        device=device,
        dtype=dtype
    )   

