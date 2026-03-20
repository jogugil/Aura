# micro_inference.py

import torch
from typing import Any, Dict, Optional, Tuple
from ..states.micro_state import MicroNeuronState, normalize_text

# ----------------------------------------------------------------------
# Funciones de activación (vectorizadas)
# ----------------------------------------------------------------------

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-x))

def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)

def tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


# ----------------------------------------------------------------------
# Activación de micro‑neuronas (vectorizada)
# ----------------------------------------------------------------------
def activate(state: MicroNeuronState,
            input_vectors: torch.Tensor,   # (T, dim) o (dim,)
            original_phrase: Optional[str] = None,
            threshold: Optional[float] = None,
            activation_fn=sigmoid) -> Tuple[MicroNeuronState, torch.Tensor]:
    """
    Actualiza el estado de activación de todas las micro‑neuronas.

    Args:
        state: estado actual.
        input_vectors: tensor de embeddings de entrada. Puede ser 1D (dim,) o 2D (T, dim).
        original_phrase: texto original (para activación por coincidencia de nombre).
        threshold: umbral global (si es None, se usa el umbral propio de cada neurona).
        activation_fn: función de activación aplicada a la activación inicial.

    Returns:
        (nuevo estado, tensor booleano de neuronas activas)
    """
    # Asegurar forma (T, dim)
    if input_vectors.dim() == 1:
        input_vectors = input_vectors.unsqueeze(0)
    T, dim = input_vectors.shape
    n = state.n
    device = state.device

    # Inicializar activación con 0
    activation = torch.zeros(n, device=device, dtype=state.dtype)

    # 1. Activación por nombre (si hay original_phrase) - no vectorizable, se hace con Python
    if original_phrase:
        norm_phrase = normalize_text(original_phrase)
        for i, concept in enumerate(state.concepts):
            if normalize_text(concept) in norm_phrase:
                activation[i] = 1.0  # máxima activación

    # 2. Activación por similitud de embeddings (vectorizada)
    # Calcular similitud coseno entre cada vector de entrada y cada neurona
    # input_vectors: (T, dim); embeddings: (n, dim) -> sim: (T, n)
    emb = state.embeddings  # (n, dim)
    # Normalizar para similitud coseno
    emb_norm = emb / (torch.norm(emb, dim=1, keepdim=True) + 1e-8)
    input_norm = input_vectors / (torch.norm(input_vectors, dim=1, keepdim=True) + 1e-8)
    sim = torch.mm(input_norm, emb_norm.t())  # (T, n)
    max_sim, _ = torch.max(sim, dim=0)          # (n,)

    # Tomar el máximo entre activación por nombre y por similitud
    activation = torch.maximum(activation, max_sim)

    # Aplicar función de activación
    activated = activation_fn(activation)

    # Determinar umbral
    if threshold is None:
        thr = state.activation_threshold
    else:
        thr = torch.full_like(activated, threshold)

    active = activated >= thr

    # Actualizar estado
    new_state = state
    new_state.activation_level = activated
    new_state.active = active
   
    # --- Registrar en historial ---
    for i in range(n):
        state.activation_history[i].append((state.activation_level[i].item(), state.active[i].item()))
        # Opcional: limitar tamaño del historial (por ejemplo, últimos 100)
        if len(state.activation_history[i]) > 100:
            state.activation_history[i] = state.activation_history[i][-100:]
    
    return new_state, active


# ----------------------------------------------------------------------
# Reset (vectorizado)
# ----------------------------------------------------------------------
def reset(state: MicroNeuronState) -> MicroNeuronState:
    """Resetea activaciones y confianza."""
    new_state = state
    new_state.activation_level.zero_()
    new_state.active.zero_()
    new_state.confidence.fill_(1.0)
    # Opcional: limpiar historial (no se hace aquí)
    return new_state

# ----------------------------------------------------------------------
# Decaimiento de activaciones (versión simple)
# ----------------------------------------------------------------------
def apply_decay(state: MicroNeuronState) -> MicroNeuronState:
    """
    Aplica decaimiento simple a las activaciones según decay_rate.
    Versión simple para usar en cognitive_engine.
    """
    state.activation_level = torch.maximum(
        torch.zeros_like(state.activation_level),
        state.activation_level - state.decay_rate
    )
    state.active = state.activation_level >= state.activation_threshold
    return state

# ----------------------------------------------------------------------
# Decaimiento contextual (versión avanzada)
# ----------------------------------------------------------------------
def apply_contextual_decay(state: MicroNeuronState, window: int = 10) -> MicroNeuronState:
    """
    Aplica decaimiento no lineal/contextual basado en el historial de activaciones.
    Si la neurona ha estado activa recientemente, el decaimiento es menor.
    """
    n = state.n
    # Calculamos la frecuencia de activación en la ventana para cada neurona
    reinforcement = torch.ones(n, device=state.device, dtype=state.dtype)
    for i in range(n):
        hist = state.activation_history[i]
        if hist:
            recent = hist[-window:]
            freq = sum(1 for h in recent if h[1]) / len(recent)
            reinforcement[i] = 1.0 + 0.5 * freq

    decay = state.decay_rate / reinforcement
    new_level = torch.maximum(
        torch.zeros_like(state.activation_level),
        state.activation_level - decay
    )
    new_state = state
    new_state.activation_level = new_level
    new_state.active = new_level >= state.activation_threshold
    return new_state 


# ----------------------------------------------------------------------
# Aprendizaje de embeddings (actualización hebbiana)
# ----------------------------------------------------------------------
def update_embedding(state: MicroNeuronState,
                         neuron_idx: int,
                         input_embedding: torch.Tensor,
                         learning_rate: float = 0.01) -> MicroNeuronState:
    """
    Actualiza el embedding de una micro‑neurona acercándolo al embedding de entrada.
    Útil para que palabras que aparecen en contextos similares tengan embeddings cercanos.
    """
    # Calcular dirección de ajuste
    direction = input_embedding - state.embeddings[neuron_idx]
    # Actualizar embedding
    state.embeddings[neuron_idx] += learning_rate * direction
    # Renormalizar para mantener magnitud estable
    state.embeddings[neuron_idx] = torch.nn.functional.normalize(
        state.embeddings[neuron_idx], dim=0
    )
    print(f"[DEBUG APRENDIZAJE] Actualizando embedding de neurona {state.ids[neuron_idx]} (concepto: {state.concepts[neuron_idx]}) con tasa {learning_rate}")
    return state

def adjust_micro_thresholds(state: MicroNeuronState, window: int = 20) -> MicroNeuronState:
    """
    Ajusta los umbrales de activación de las micro‑neuronas según su frecuencia reciente.
    Homeostasis: si una neurona se activa demasiado, sube su umbral; si muy poco, lo baja.
    """
    for i in range(state.n):
        hist = state.activation_history[i]
        if len(hist) >= window:
            recent = hist[-window:]
            # Cada entrada es (activation_level, active) – tomamos el segundo (active)
            activation_rate = sum(1 for h in recent if h[1]) / window
            if activation_rate > 0.7:
                state.activation_threshold[i] = min(1.0, state.activation_threshold[i] + 0.02)
            elif activation_rate < 0.1:
                state.activation_threshold[i] = max(0.1, state.activation_threshold[i] - 0.02)
            
            # Mostrar mensajes de depuración
            if activation_rate > 0.7:
                print(f"[DEBUG HOMEOSTASIS] Micro {state.ids[i]} umbral sube a {state.activation_threshold[i].item():.2f} (tasa {activation_rate:.2f})")
            elif activation_rate < 0.1:
                print(f"[DEBUG HOMEOSTASIS] Micro {state.ids[i]} umbral baja a {state.activation_threshold[i].item():.2f} (tasa {activation_rate:.2f})")
    return state

# ----------------------------------------------------------------------
# Similitud coseno (vectorizada entre dos conjuntos)
# ----------------------------------------------------------------------
def cosine_similarity_batch(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
    """
    Calcula la similitud coseno entre cada vector de emb1 y cada vector de emb2.
    emb1: (n1, dim)
    emb2: (n2, dim)
    Retorna: (n1, n2)
    """
    emb1_norm = emb1 / (torch.norm(emb1, dim=1, keepdim=True) + 1e-8)
    emb2_norm = emb2 / (torch.norm(emb2, dim=1, keepdim=True) + 1e-8)
    return torch.mm(emb1_norm, emb2_norm.t())

# ----------------------------------------------------------------------
# Obtención de datos para índice (compatibilidad)
# ----------------------------------------------------------------------
def get_index_data(state: MicroNeuronState, idx: int) -> Tuple[str, torch.Tensor, Dict[str, Any]]:
    """Devuelve (id, embedding, metadata) de la neurona índice idx."""
    return state.ids[idx], state.embeddings[idx], state.metadata[idx]



