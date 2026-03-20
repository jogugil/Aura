# neuron_inference.py

import torch
from typing import Optional, Tuple
from collections import defaultdict
from ..states.neuron_state import NeuronState

# ----------------------------------------------------------------------
# Funciones de activación (reutilizamos las de micro_neuron)
# ----------------------------------------------------------------------
def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-x))

def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)

def tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


# ----------------------------------------------------------------------
# Evaluación de todas las neuronas
# ----------------------------------------------------------------------
def evaluate_neurons(
    state: NeuronState,
    micro_activations: torch.Tensor,          # (M,) niveles de activación de micro‑neuronas
    global_threshold: Optional[float] = None,
    activation_fn=sigmoid,
    beta_frequency: float = 0.9,              # EMA para frecuencia
    use_attention: bool = True,
    learning_rate: float = 0.05,               # para actualización Hebbiana
    lambda_decay: float = 0.01,           # decaimiento de pesos
    learning_threshold: float = 0.85,           # 🌟 Solo neuronas con activación > esto aprenden
    micro_learning_threshold: float = 0.7       # 🌟 Solo micros con activación > esto contribuyen
) -> Tuple[NeuronState, torch.Tensor]:
    """
    Evalúa todas las neuronas dado el estado de micro‑neuronas.
    Actualiza in‑place el estado de las neuronas (activation_level, active)
    y también actualiza la frecuencia de micro‑neuronas y los pesos (Hebb).
    
    El aprendizaje Hebbiano ahora es LOCAL, ESCASO y SELECTIVO:
    - Solo se actualizan neuronas con activación > learning_threshold
    - Solo se actualizan conexiones donde la micro pre-sináptica tiene activación > micro_learning_threshold
    - El resto de conexiones permanecen inalteradas

    Args:
        state: estado de las neuronas
        micro_activations: tensor (M,) con niveles de activación de micro‑neuronas
        global_threshold: si se proporciona, se usa este umbral para todas
        activation_fn: función de activación
        beta_frequency: factor de suavizado para el EMA de frecuencia
        use_attention: si se aplica atención contextual basada en frecuencia
        learning_rate: tasa de aprendizaje para Hebb
        lambda_decay: factor de decaimiento de pesos
        learning_threshold: 🌟 umbral para considerar que una neurona debe aprender
        micro_learning_threshold: 🌟 umbral para considerar que una micro contribuye al aprendizaje

    Returns:
        el mismo estado (modificado) y un tensor booleano con las neuronas activas
    """
    N = state.N
    M = state.M
    device = state.device

    # Asegurar que micro_activations esté en el dispositivo correcto
    micro_activations = micro_activations.to(device)

    # ------------------------------------------------
    # 1. Lógica de exclusión
    # ------------------------------------------------
    if state.exclusions._nnz() > 0:
        excl_sum = torch.sparse.mm(state.exclusions, micro_activations.unsqueeze(1)).squeeze(1)
        excluded = excl_sum > 0
    else:
        excluded = torch.zeros(N, dtype=torch.bool, device=device)

    # ------------------------------------------------
    # 2. Atención contextual (frecuencia)
    # ------------------------------------------------
    if use_attention:
        active_micro = (micro_activations > 0).float()
        state.micro_frequency = beta_frequency * state.micro_frequency + (1 - beta_frequency) * active_micro

        attention = 0.5 + 0.5 * state.micro_frequency
        weights_dense = state.weights.to_dense()
        weights_with_attention = weights_dense * attention.unsqueeze(0)
        weighted_sum = torch.mv(weights_with_attention, micro_activations)
    else:
        weighted_sum = torch.sparse.mm(state.weights, micro_activations.unsqueeze(1)).squeeze(1)

    # ------------------------------------------------
    # 3. Aplicar función de activación y umbral
    # ------------------------------------------------
    activated = activation_fn(weighted_sum)
    activated[excluded] = 0.0

    if global_threshold is not None:
        threshold = torch.full_like(activated, global_threshold)
    else:
        threshold = state.activation_threshold

    active = activated >= threshold

    # Actualizar historial
    for i in range(N):
        state.activation_history[i].append((state.activation_level[i].item(), state.active[i].item()))
    
    # Actualizar estado
    state.activation_level = activated
    state.active = active

    # ============================================================
    # 4. APRENDIZAJE HEBBIANO SELECTIVO (MODIFICADO)
    # ============================================================
    if learning_rate != 0:
        # --------------------------------------------------------
        # 4.1 Identificar neuronas candidatas a aprendizaje
        #     (solo aquellas con activación > learning_threshold)
        # --------------------------------------------------------
        candidate_neurons = (state.activation_level > learning_threshold).nonzero(as_tuple=True)[0]
        
        if len(candidate_neurons) > 0:
            # ----------------------------------------------------
            # 4.2 Identificar micros candidatas (pre-sinápticas)
            #     (solo aquellas con activación > micro_learning_threshold)
            # ----------------------------------------------------
            candidate_micros = (micro_activations > micro_learning_threshold).nonzero(as_tuple=True)[0]
            
            if len(candidate_micros) > 0:
                # ------------------------------------------------
                # 4.3 Obtener índices de todas las conexiones existentes
                # ------------------------------------------------
                indices = state.weights._indices()      # (2, nnz)
                values = state.weights._values()       # (nnz,)
                
                # ------------------------------------------------
                # 4.4 Crear máscara para conexiones que cumplen:
                #     - La neurona post-sináptica es candidata
                #     - La micro pre-sináptica es candidata
                # ------------------------------------------------
                # Máscara para neuronas candidatas
                mask_neurons = torch.isin(indices[0], candidate_neurons)
                
                # Máscara para micros candidatas
                mask_micros = torch.isin(indices[1], candidate_micros)
                
                # Combinar máscaras: solo conexiones donde AMBAS son candidatas
                learning_mask = mask_neurons & mask_micros
                
                # ------------------------------------------------
                # 4.5 Si hay conexiones que cumplen, actualizarlas
                # ------------------------------------------------
                if learning_mask.any():
                    # Calcular delta Hebbiano solo para esas conexiones
                    # delta = learning_rate * (act_post * act_pre)
                    delta = learning_rate * (
                        state.activation_level[indices[0][learning_mask]] * 
                        micro_activations[indices[1][learning_mask]]
                    )
                    
                    # Crear copia de valores para actualizar
                    new_values = values.clone()
                    
                    # Aplicar delta solo a las conexiones seleccionadas
                    new_values[learning_mask] += delta
                    
                    # Aplicar decaimiento a TODAS las conexiones
                    new_values = new_values - lambda_decay * new_values
                    
                    # Podar valores muy pequeños
                    pruning_mask = new_values.abs() > 1e-6
                    new_indices = indices[:, pruning_mask]
                    new_values = new_values[pruning_mask]
                    
                    # Reconstruir tensor disperso
                    state.weights = torch.sparse_coo_tensor(
                        new_indices, 
                        new_values, 
                        (N, M), 
                        device=device
                    )
                else:
                    # No hay conexiones que cumplan los criterios, solo aplicar decaimiento
                    new_values = values - lambda_decay * values
                    pruning_mask = new_values.abs() > 1e-6
                    new_indices = indices[:, pruning_mask]
                    new_values = new_values[pruning_mask]
                    state.weights = torch.sparse_coo_tensor(
                        new_indices, 
                        new_values, 
                        (N, M), 
                        device=device
                    )
            else:
                # No hay micros candidatas, solo aplicar decaimiento
                indices = state.weights._indices()
                values = state.weights._values()
                new_values = values - lambda_decay * values
                pruning_mask = new_values.abs() > 1e-6
                new_indices = indices[:, pruning_mask]
                new_values = new_values[pruning_mask]
                state.weights = torch.sparse_coo_tensor(
                    new_indices, 
                    new_values, 
                    (N, M), 
                    device=device
                )
        else:
            # No hay neuronas candidatas, solo aplicar decaimiento
            indices = state.weights._indices()
            values = state.weights._values()
            new_values = values - lambda_decay * values
            pruning_mask = new_values.abs() > 1e-6
            new_indices = indices[:, pruning_mask]
            new_values = new_values[pruning_mask]
            state.weights = torch.sparse_coo_tensor(
                new_indices, 
                new_values, 
                (N, M), 
                device=device
            )

    return state, active

def lateral_inhibition(state: NeuronState, factor: float = 0.7) -> NeuronState:
    """
    Reduce la activación de neuronas con el mismo conjunto de condiciones (solapamiento).
    Se agrupan por el conjunto de condiciones (usando un hash) y se reduce la activación
    de las que no son la máxima del grupo.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for i, conds in enumerate(state.condition_indices):
        key = tuple(sorted(conds))
        groups[key].append(i)

    for group in groups.values():
        if len(group) > 1:
            acts = state.activation_level[group]
            max_act = acts.max()
            mask = acts < max_act
            indices_to_reduce = [group[j] for j, m in enumerate(mask) if m]
            if indices_to_reduce:
                state.activation_level[indices_to_reduce] *= factor
                # Recalcular activas
                state.active[indices_to_reduce] = state.activation_level[indices_to_reduce] >= state.activation_threshold[indices_to_reduce]
    return state

# ----------------------------------------------------------------------
# Decaimiento de activaciones (vectorizado)
# ----------------------------------------------------------------------
def apply_neuron_decay(state: NeuronState) -> NeuronState:
    """Reduce las activaciones según decay_rate."""
    state.activation_level = torch.maximum(
        torch.zeros_like(state.activation_level),
        state.activation_level - state.decay_rate
    )
    state.active = state.activation_level >= state.activation_threshold
    return state


# ----------------------------------------------------------------------
# Reset
# ----------------------------------------------------------------------
def reset_neurons(state: NeuronState) -> NeuronState:
    """Pone a cero las activaciones."""
    state.activation_level.zero_()
    state.active.zero_()
    return state




def adjust_neuron_thresholds(state: NeuronState, window: int = 20) -> NeuronState:
    """
    Ajusta los umbrales de activación de cada neurona usando un mecanismo de homeostasis.

    Idea principal:
    - Si una neurona se activa demasiado, se aumenta su umbral (más difícil activarse)
    - Si se activa muy poco, se reduce su umbral (más fácil activarse)

    Parámetros:
    - state: objeto que contiene el estado de las neuronas
    - window: número de pasos recientes a considerar para el cálculo

    Devuelve:
    - state actualizado con los nuevos umbrales
    """

    # Evita valores inválidos de ventana
    if window <= 0:
        return state

    # Recorre todas las neuronas
    for i in range(state.N):

        # Comprueba que existe historial para esta neurona
        if i >= len(state.activation_history):
            continue

        hist = state.activation_history[i]

        # Si no hay historial o no hay suficientes datos, se salta
        if not hist or len(hist) < window:
            continue

        # Toma las últimas 'window' entradas del historial
        recent = hist[-window:]

        activation_count = 0  # número de veces que la neurona estuvo activa
        valid_count = 0       # número de entradas válidas

        # Analiza cada entrada del historial reciente
        for h in recent:
            # Verifica que la entrada tenga la estructura esperada
            if isinstance(h, (list, tuple)) and len(h) > 1:
                valid_count += 1

                # Convierte el valor a booleano por seguridad
                if bool(h[1]):
                    activation_count += 1

        # Si no hay datos válidos, no se hace nada
        if valid_count == 0:
            continue

        # Calcula la tasa de activación
        activation_rate = activation_count / valid_count

        # Obtiene el valor del umbral, compatible con tensor o float
        threshold = state.activation_threshold[i]
        threshold_val = threshold.item() if hasattr(threshold, "item") else float(threshold)

        # Si la neurona se activa demasiado, se aumenta el umbral
        if activation_rate > 0.7:
            state.activation_threshold[i] = min(1.0, threshold_val + 0.02)

        # Si se activa muy poco, se reduce el umbral
        elif activation_rate < 0.1:
            state.activation_threshold[i] = max(0.1, threshold_val - 0.02)

        # Si está en un rango intermedio, no se modifica

    return state