import os
import logging
import torch
from gensim.models import KeyedVectors

logger = logging.getLogger(__name__)

class WordVectors:
    """
    @inproceedings{grave2018learning,
        title={Learning Word Vectors for 157 Languages},
        author={Grave, Edouard and Bojanowski, Piotr and Gupta, Prakhar and Joulin, Armand and Mikolov, Tomas},
        booktitle={Proceedings of the International Conference on Language Resources and Evaluation (LREC 2018)},
        year={2018}
    }
    """
    def __init__(self, path):
        self.model = None
        self.dim = None
        self._load_vectors(path)

    def _get_binary_path(self, text_path: str) -> str:
        """Genera la ruta para el archivo binario a partir del texto."""
        return text_path + '.bin'

    def _load_vectors(self, path: str):
        """Carga los vectores FastText, usando caché binario si existe."""
        bin_path = self._get_binary_path(path)
        
        # Si existe el binario, cargar desde ahí (mucho más rápido)
        if os.path.exists(bin_path):
            logger.info(f"Cargando vectores desde binario: {bin_path}")
            self.model = KeyedVectors.load(bin_path)
            self.dim = self.model.vector_size
            logger.info(f"✓ Vectores cargados desde binario. Dimensión: {self.dim}")
            return
        
        # Si no, cargar desde texto (lento) y luego guardar binario
        logger.info(f"Cargando vectores FastText desde texto: {path} (puede tardar unos segundos)...")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Archivo no encontrado: {path}")
        
        self.model = KeyedVectors.load_word2vec_format(path, binary=False)
        self.dim = self.model.vector_size
        logger.info(f"✓ Vectores cargados. Dimensión: {self.dim}")
        
        # Guardar versión binaria para futuras cargas
        logger.info(f"Guardando versión binaria en {bin_path} para futuras cargas...")
        self.model.save(bin_path)
        logger.info("Versión binaria guardada.")
    
    def get_word_vector(self, word: str) -> torch.Tensor:
        """Devuelve vector de palabra como tensor PyTorch."""
        if self.model is None:
            raise RuntimeError("FastText no cargado.")
        try:
            return torch.tensor(self.model[word], dtype=torch.float32)
        except KeyError:
            return torch.zeros(self.dim, dtype=torch.float32)

    def embed_text(self, text: str) -> torch.Tensor:
        """Embedding de frase como promedio de vectores (tensor)."""
        words = text.lower().split()
        if not words:
            return torch.zeros(self.dim, dtype=torch.float32)
        
        vectors = [self.get_word_vector(w) for w in words]
        stacked = torch.stack(vectors)  # (n_palabras, dim)
        return stacked.mean(dim=0)


# Singleton
_word_vectors = None

def get_word_vectors(path="data/cc.es.300.vec"):
    global _word_vectors
    if _word_vectors is None:
        _word_vectors = WordVectors(path)
    return _word_vectors