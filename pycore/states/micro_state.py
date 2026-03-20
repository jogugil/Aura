import torch
# import math
import unicodedata
# import torch.nn.functional as F
 
from typing import List, Dict, Any, Optional, Tuple

# ----------------------------------------------------------------------
# Funciones auxiliares de procesamiento de texto (no jit, se usan solo al crear)
# ----------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Normaliza un texto (minúsculas, elimina tildes, caracteres no alfanuméricos)."""
    text = text.lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    return text.strip()

def generate_ngrams(word: str, min_n: int = 2, max_n: int = 5) -> List[str]:
    """Genera n-grams de caracteres para una palabra."""
    ngrams = {word}
    for i in range(1, min(len(word), max_n + 1)):
        ngrams.add(word[:i])
        ngrams.add(word[-i:])
    for n in range(min_n, max_n + 1):
        for i in range(len(word) - n + 1):
            ngrams.add(word[i:i+n])
    return list(ngrams)

def compute_embedding(text: str, dim: int = 64) -> torch.Tensor:
    """
    Calcula un embedding determinista para un texto.
    Se basa en n-grams y una semilla hash. Devuelve un tensor 1D normalizado.
    """
    norm_text = normalize_text(text)
    if not norm_text:
        return torch.zeros(dim)
    ngrams = generate_ngrams(norm_text)
    vec = torch.zeros(dim)
    for ng in ngrams:
        # Semilla determinista a partir del hash
        seed = hash(ng) % 2**32
        torch.manual_seed(seed)
        # Generar vector aleatorio uniforme en [-1, 1]
        contrib = torch.empty(dim).uniform_(-1, 1)
        vec += contrib
    norm = torch.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

# ----------------------------------------------------------------------
# Estado global de micro‑neuronas (tensores agrupados)
# ----------------------------------------------------------------------
class MicroNeuronState:
    """
    Contiene todos los datos de un conjunto de micro‑neuronas en forma de tensores.
    Los metadatos que no son tensoriales se guardan en listas paralelas de Python.
    """
    def __init__(self,
                 ids: List[str],
                 concepts: List[str],
                 types: List[str],
                 embeddings: Optional[torch.Tensor] = None,
                 metadata: Optional[List[Dict[str, Any]]] = None,
                 activation_threshold: float = 0.7,
                 decay_rate: float = 0.15,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device('cpu')):
        """
        Crea el estado para N micro‑neuronas.
        Si embeddings es None, se calculan automáticamente a partir de concepts.
        """
        self.device = device
        self.dtype = dtype
        self.n = len(ids)
        self.ids = ids
        self.concepts = concepts
        self.types = types
        if metadata is None:
            metadata = [{} for _ in range(self.n)]
        self.metadata = metadata

        # Embeddings: (n, dim) tensor
        if embeddings is None:
            # Calcular embeddings secuencialmente (solo una vez, no en el bucle de entrenamiento)
            emb_list = []
            for c in concepts:
                emb = compute_embedding(c, dim=64)
                emb_list.append(emb)
            self.embeddings = torch.stack(emb_list).to(device)
        else:
            self.embeddings = embeddings.to(device)

        # Parámetros escalares por neurona (tensores)
        self.activation_threshold = torch.full((self.n,), activation_threshold, device=device, dtype=dtype)
        self.decay_rate = torch.full((self.n,), decay_rate, device=device, dtype=dtype)

        # Estado variable
        self.activation_level = torch.zeros(self.n, device=device, dtype=dtype)
        self.active = torch.zeros(self.n, dtype=torch.bool, device=device)
        self.confidence = torch.ones(self.n, device=device, dtype=dtype)

        # Historial (no se vectoriza, se mantiene como listas Python)
        self.activation_history = [[] for _ in range(self.n)]

        # Para caché de similitudes (opcional, se puede implementar después)
        self.similarity_cache = {}  # simple diccionario Python

    def to(self, device: torch.device):
        """Mueve todos los tensores a otro dispositivo."""
        self.device = device
        self.embeddings = self.embeddings.to(device)
        self.activation_threshold = self.activation_threshold.to(device)
        self.decay_rate = self.decay_rate.to(device)
        self.activation_level = self.activation_level.to(device)
        self.active = self.active.to(device)
        self.confidence = self.confidence.to(device)
        return self
    
   



