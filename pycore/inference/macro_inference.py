# macro_inference.py

import torch
from typing import Tuple
from ..states.macro_state import MacroNeuronState


def evaluate_macro_neurons(
state: MacroNeuronState,
active_neuron_indices: torch.Tensor,      # (K,) índices de neuronas activas (nivel intermedio)
active_micro_indices: torch.Tensor          # (L,) índices de micro‑neuronas activas
) -> Tuple[MacroNeuronState, torch.Tensor]:   # devuelve (estado, activas)
    """
    Evalúa todas las macro‑neuronas dadas las neuronas y micro‑neuronas activas.
    Actualiza in‑place el estado.
    """
    device = state.device

    # Asegurar que los índices están en el dispositivo correcto
    active_neuron_indices = active_neuron_indices.to(device)
    active_micro_indices = active_micro_indices.to(device)

    # --- 1. Exclusión por micro‑neuronas ---
    # Vector indicador de micro‑neuronas activas (1 donde activas)
    if state.Nmicro > 0:
        active_micro_vec = torch.zeros(state.Nmicro, device=device, dtype=state.dtype)
        if len(active_micro_indices) > 0:
            active_micro_vec.scatter_(0, active_micro_indices, 1.0)
    else:
        active_micro_vec = torch.tensor([], device=device, dtype=state.dtype)

    # Calcular para cada macro si tiene alguna exclusión activa
    if state.exclusions._nnz() > 0 and state.Nmicro > 0:
        excl_sum = torch.sparse.mm(state.exclusions, active_micro_vec.unsqueeze(1)).squeeze(1)   # (Nm,)
        excluded = excl_sum > 0
    else:
        excluded = torch.zeros(state.Nm, dtype=torch.bool, device=device)

    # --- 2. Evaluación por condiciones de neuronas ---
    # Vector indicador de neuronas activas
    if state.Nn > 0:
        active_neuron_vec = torch.zeros(state.Nn, device=device, dtype=state.dtype)
        if len(active_neuron_indices) > 0:
            active_neuron_vec.scatter_(0, active_neuron_indices, 1.0)
    else:
        active_neuron_vec = torch.tensor([], device=device, dtype=state.dtype)

    # Contar condiciones activas por macro
    if state.conditions._nnz() > 0 and state.Nn > 0:
        active_conditions_count = torch.sparse.mm(state.conditions, active_neuron_vec.unsqueeze(1)).squeeze(1)  # (Nm,)
    else:
        active_conditions_count = torch.zeros(state.Nm, device=device, dtype=state.dtype)

    # Calcular proporción (evitar división por cero)
    no_condition_mask = (state.condition_lengths == 0)
    proportion = torch.where(
        no_condition_mask,
        torch.zeros_like(active_conditions_count),
        active_conditions_count / state.condition_lengths.float()
    )

    # Aplicar umbral
    active_by_condition = proportion >= state.threshold

    # Combinar exclusión y condiciones
    active = active_by_condition & (~excluded)

    # Actualizar estado
    state.active = active
    state.activation_level = proportion   # guardamos la proporción como nivel continuo

    return state, active


def reset_macro_neurons(state: MacroNeuronState) -> MacroNeuronState:
    """Pone a cero las activaciones."""
    state.active.zero_()
    state.activation_level.zero_()
    return state

