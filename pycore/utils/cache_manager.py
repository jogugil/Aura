"""
Sistema de Caché Inteligente para Krystal AI
Optimiza el rendimiento mediante caché LRU de similitudes y activaciones.
Versión adaptada para trabajar con tensores PyTorch (las claves se generan con pickle, que maneja tensores correctamente).
"""

import time
import threading
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional, Any
import hashlib
import pickle


class LRUCache:
    """Implementación de caché LRU thread-safe con TTL opcional."""
    
    def __init__(self, max_size: int = 1000, ttl: Optional[float] = None):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()
        self.timestamps = {}
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def _is_expired(self, key: str) -> bool:
        if self.ttl is None:
            return False
        return time.time() - self.timestamps.get(key, 0) > self.ttl
    
    def _cleanup_expired(self):
        if self.ttl is None:
            return
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.timestamps.items()
            if current_time - timestamp > self.ttl
        ]
        for key in expired_keys:
            self.cache.pop(key, None)
            self.timestamps.pop(key, None)
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key not in self.cache or self._is_expired(key):
                self.misses += 1
                if key in self.cache:
                    del self.cache[key]
                    del self.timestamps[key]
                return None
            # Mover al final (más reciente)
            value = self.cache.pop(key)
            self.cache[key] = value
            self.hits += 1
            return value
    
    def put(self, key: str, value: Any):
        with self.lock:
            if len(self.cache) % 100 == 0:
                self._cleanup_expired()
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'ttl': self.ttl
            }


class CacheManager:
    """Gestor central de caché para el sistema Krystal AI."""
    
    def __init__(self):
        self.similarity_cache = LRUCache(max_size=10000, ttl=3600)   # 1 hora
        self.embedding_cache = LRUCache(max_size=5000, ttl=7200)     # 2 horas
        self.activation_cache = LRUCache(max_size=2000, ttl=1800)    # 30 min
        self.evaluation_cache = LRUCache(max_size=1000, ttl=1800)    # 30 min
        self.start_time = time.time()
    
    def _generate_key(self, *args) -> str:
        """Genera una clave única a partir de argumentos serializables (soporta tensores PyTorch)."""
        serialized = pickle.dumps(args, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.md5(serialized).hexdigest()
    
    def get_similarity(self, vector1, vector2) -> Optional[float]:
        """Obtiene similitud coseno del caché. Acepta listas o tensores."""
        # Convertir tensores a listas para consistencia (o podríamos usar el tensor directamente en pickle)
        if hasattr(vector1, 'tolist'):
            vector1 = vector1.tolist()
        if hasattr(vector2, 'tolist'):
            vector2 = vector2.tolist()
        key = self._generate_key('similarity', tuple(vector1), tuple(vector2))
        return self.similarity_cache.get(key)
    
    def cache_similarity(self, vector1, vector2, similarity: float):
        if hasattr(vector1, 'tolist'):
            vector1 = vector1.tolist()
        if hasattr(vector2, 'tolist'):
            vector2 = vector2.tolist()
        key = self._generate_key('similarity', tuple(vector1), tuple(vector2))
        self.similarity_cache.put(key, similarity)
        # También la simetría
        key_inv = self._generate_key('similarity', tuple(vector2), tuple(vector1))
        self.similarity_cache.put(key_inv, similarity)
    
    def get_embedding(self, text: str, dim: int) -> Optional[List[float]]:
        key = self._generate_key('embedding', text, dim)
        return self.embedding_cache.get(key)
    
    def cache_embedding(self, text: str, dim: int, embedding):
        # embedding puede ser lista o tensor; guardamos como lista para ahorrar espacio?
        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()
        key = self._generate_key('embedding', text, dim)
        self.embedding_cache.put(key, embedding)
    
    def get_activation(self, micro_id: str, input_vectors, original_phrase: str, threshold: float) -> Optional[Tuple[bool, float]]:
        # input_vectors puede ser lista de tensores o listas
        vecs = []
        for v in input_vectors:
            if hasattr(v, 'tolist'):
                v = v.tolist()
            vecs.append(tuple(v))
        key = self._generate_key('activation', micro_id, vecs, original_phrase, threshold)
        return self.activation_cache.get(key)
    
    def cache_activation(self, micro_id: str, input_vectors, original_phrase: str, threshold: float, result: Tuple[bool, float]):
        vecs = []
        for v in input_vectors:
            if hasattr(v, 'tolist'):
                v = v.tolist()
            vecs.append(tuple(v))
        key = self._generate_key('activation', micro_id, vecs, original_phrase, threshold)
        self.activation_cache.put(key, result)
    
    def get_evaluation(self, neuron_id: str, active_concepts: Dict[str, float]) -> Optional[Tuple[bool, float]]:
        items = tuple(sorted(active_concepts.items()))
        key = self._generate_key('evaluation', neuron_id, items)
        return self.evaluation_cache.get(key)
    
    def cache_evaluation(self, neuron_id: str, active_concepts: Dict[str, float], result: Tuple[bool, float]):
        items = tuple(sorted(active_concepts.items()))
        key = self._generate_key('evaluation', neuron_id, items)
        self.evaluation_cache.put(key, result)
    
    def invalidate_neuron_caches(self, neuron_id: str):
        # Simplificación: limpiar cachés completos
        self.activation_cache.clear()
        self.evaluation_cache.clear()
    
    def get_global_stats(self) -> Dict[str, Any]:
        uptime = time.time() - self.start_time
        return {
            'uptime_seconds': uptime,
            'similarity_cache': self.similarity_cache.get_stats(),
            'embedding_cache': self.embedding_cache.get_stats(),
            'activation_cache': self.activation_cache.get_stats(),
            'evaluation_cache': self.evaluation_cache.get_stats(),
            'total_memory_entries': (
                len(self.similarity_cache.cache) +
                len(self.embedding_cache.cache) +
                len(self.activation_cache.cache) +
                len(self.evaluation_cache.cache)
            )
        }
    
    def clear_all_caches(self):
        self.similarity_cache.clear()
        self.embedding_cache.clear()
        self.activation_cache.clear()
        self.evaluation_cache.clear()
    
    def optimize_memory(self):
        self.activation_cache._cleanup_expired()
        self.evaluation_cache._cleanup_expired()
        self.similarity_cache._cleanup_expired()
        self.embedding_cache._cleanup_expired()


# Instancia global del gestor de caché
cache_manager = CacheManager() 