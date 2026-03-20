import torch
import logging
from typing import List, Dict, Any, Optional, Tuple

# Configurar logger
logger = logging.getLogger(__name__)

# ============================================================================
# CAPA 1: ESTADO PURO (solo datos, sin lógica)
# ============================================================================

class MacroNeuronState:
    """
    Contiene todas las macro‑neuronas en forma de tensores.
    - Nm: número de macro‑neuronas
    - Nn: número total de neuronas (de nivel intermedio)
    - Nmicro: número total de micro‑neuronas
    """
    def __init__(self,
                 ids: List[str],
                 names: List[str],
                 condition_indices: List[List[int]],   # lista de listas de índices de neuronas que activan
                 exclusion_indices: List[List[int]],   # lista de listas de índices de micro‑neuronas que excluyen
                 threshold: float = 0.5,
                 device: torch.device = torch.device('cpu'),
                 dtype: torch.dtype = torch.float32):
        """
        Args:
            ids: identificadores de las macro‑neuronas
            names: nombres descriptivos
            condition_indices: para cada macro, índices de las neuronas que la activan
            exclusion_indices: para cada macro, índices de las micro‑neuronas que la excluyen
            threshold: umbral por defecto (escalar)
        """
        self.device = device
        self.dtype = dtype
        self.Nm = len(ids)

        # Determinar dimensiones
        max_cond = max([max(cond, default=-1) for cond in condition_indices]) + 1 if condition_indices else 0
        max_excl = max([max(excl, default=-1) for excl in exclusion_indices]) + 1 if exclusion_indices else 0
        self.Nn = max_cond
        self.Nmicro = max_excl

        # Metadatos (no tensoriales)
        self.ids = ids
        self.names = names
        self.metadata = [{} for _ in range(self.Nm)]
        self.condition_indices = condition_indices   # guardar para inhibición lateral (como listas)
        self.exclusion_indices = exclusion_indices

        # Matriz de condiciones (dispersa COO) – para saber qué neuronas activan a cada macro
        cond_rows = []
        cond_cols = []
        condition_lengths = []
        for i, conds in enumerate(condition_indices):
            condition_lengths.append(len(conds))
            for j in conds:
                cond_rows.append(i)
                cond_cols.append(j)
        self.condition_lengths = torch.tensor(condition_lengths, device=device, dtype=torch.long)
        if cond_rows:
            cond_indices = torch.tensor([cond_rows, cond_cols], dtype=torch.long, device=device)
            cond_values = torch.ones(len(cond_rows), dtype=dtype, device=device)
            self.conditions = torch.sparse_coo_tensor(cond_indices, cond_values, (self.Nm, self.Nn), device=device)
        else:
            self.conditions = torch.sparse_coo_tensor(torch.zeros(2, 0, dtype=torch.long), torch.empty(0), (self.Nm, self.Nn), device=device)

        # Matriz de exclusiones (dispersa COO) – micro‑neuronas que excluyen
        excl_rows = []
        excl_cols = []
        for i, exc in enumerate(exclusion_indices):
            for j in exc:
                excl_rows.append(i)
                excl_cols.append(j)
        if excl_rows:
            excl_indices = torch.tensor([excl_rows, excl_cols], dtype=torch.long, device=device)
            excl_values = torch.ones(len(excl_rows), dtype=dtype, device=device)
            self.exclusions = torch.sparse_coo_tensor(excl_indices, excl_values, (self.Nm, self.Nmicro), device=device)
        else:
            self.exclusions = torch.sparse_coo_tensor(torch.zeros(2, 0, dtype=torch.long), torch.empty(0), (self.Nm, self.Nmicro), device=device)

        # Umbrales (uno por macro‑neurona)
        self.threshold = torch.full((self.Nm,), threshold, device=device, dtype=dtype)

        # Estado variable
        self.active = torch.zeros(self.Nm, dtype=torch.bool, device=device)
        self.activation_level = torch.zeros(self.Nm, device=device, dtype=dtype)   # nivel continuo (proporción)

        # Historial (opcional, para compatibilidad)
        self.activation_history = [[] for _ in range(self.Nm)]

        # Matriz de transiciones entre conceptos (para multilingüismo y gramática aprendida)
        self.transitions = torch.zeros(self.Nm, self.Nm, device=device, dtype=dtype)

    def to(self, device: torch.device):
        """Mueve todos los tensores a otro dispositivo."""
        self.device = device
        self.conditions = self.conditions.to(device)
        self.exclusions = self.exclusions.to(device)
        self.condition_lengths = self.condition_lengths.to(device)
        self.threshold = self.threshold.to(device)
        self.active = self.active.to(device)
        self.activation_level = self.activation_level.to(device)
        self.transitions = self.transitions.to(device)
        return self

def create_macro_state_from_ids(
    macro_ids: List[str],
    names: List[str],
    condition_indices: List[List[int]],
    exclusion_indices: List[List[int]],
    neuron_id_to_index: Dict[str, int],
    micro_id_to_index: Dict[str, int],
    threshold: float = 0.5,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32
) -> MacroNeuronState:
    """
    Crea un estado de macro‑neuronas a partir de listas de índices.
    (Las condiciones ya vienen como índices de neuronas, y exclusiones como índices de micro)
    """
    return MacroNeuronState(
        ids=macro_ids,
        names=names,
        condition_indices=condition_indices,
        exclusion_indices=exclusion_indices,
        threshold=threshold,
        device=device,
        dtype=dtype
    )