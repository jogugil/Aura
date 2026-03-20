import yaml
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


# ===============================
# CONFIG STRUCTURES
# ===============================
@dataclass
class EmbeddingsConfig:
    dimension: int = 300
    use_fasttext: bool = True
    fallback_dimension: int = 64
    normalize: bool = True

@dataclass
class MetaAdjustConfig:
    window: int
    low_threshold: float
    high_threshold: float
    threshold_adjust: float
    decay_adjust: float


@dataclass
class EngineConfig:
    micro_input_threshold: float
    num_iterations: int
    feedback_strength: float
    meta_adjust: MetaAdjustConfig
    # CAMPOS EXISTENTES CON VALORES POR DEFECTO
    macro_neuron_threshold: float = 0.5
    macro_initial_transition: float = 0.1


@dataclass
class ThinkingConfig:
    thinking_memory_threshold: float


@dataclass
class NeuronConfig:
    use_attention: bool
    learning_rate: float
    lambda_decay: float
    # CAMPOS EXISTENTES CON VALORES POR DEFECTO
    initial_threshold: float = 0.6
    initial_decay: float = 0.25
    min_weight: float = 0.3
    max_weight: float = 1.0
    # 🌟 NUEVOS CAMPOS PARA APRENDIZAJE SELECTIVO
    learning_threshold: float = 0.85        # Solo neuronas con activación > esto aprenden
    micro_learning_threshold: float = 0.7   # Solo micros con activación > esto contribuyen


@dataclass
class ModelConfig:
    seed: int
    engine: EngineConfig
    thinking: ThinkingConfig
    neuron: NeuronConfig


@dataclass
class InhibitionConfig:
    neuron_factor: float
    micro_factor: float


@dataclass
class DecayConfig:
    micro_factor: float
    neuron_factor: float


@dataclass
class DynamicsConfig:
    inhibition: InhibitionConfig
    decay: DecayConfig


@dataclass
class ActivationConfig:
    function: str


@dataclass
class LearningConfig:
    scheduler: Optional[str]
    grad_clip: Optional[float]
    # 🌟 NUEVO: Campo para controlar si el aprendizaje Hebbiano está activado
    hebbian_enabled: bool = True


@dataclass
class RegularizationConfig:
    dropout: float
    activation_noise: float


@dataclass
class LoggingConfig:
    level: str
    save_metrics: bool
    log_path: str


# ===============================
# SECCIONES EXISTENTES: PATHS Y CLUSTERING
# ===============================

@dataclass
class PathsConfig:
    clustering_state: str = "data/clustering_state.pkl"
    # 🌟 NUEVO: Ruta para guardar/recuperar vectores FastText
    fasttext_vectors: str = "data/fasttext_vectors.vec"


@dataclass
class ClusteringConfig:
    cooccurrence_threshold: float = 0.7
    context_window: int = 10
    detection_frequency: int = 20
    # 🌟 NUEVO: Número mínimo de micros para formar un concepto
    min_words_per_concept: int = 2


@dataclass
class AppConfig:
    model: ModelConfig
    dynamics: DynamicsConfig
    activation: ActivationConfig
    learning: LearningConfig
    regularization: RegularizationConfig
    logging: LoggingConfig
    paths: PathsConfig
    clustering: ClusteringConfig
    embeddings: EmbeddingsConfig

# ===============================
# VALIDATION
# ===============================

def _validate_config(config: AppConfig):
    # Validaciones existentes
    assert 0 < config.model.engine.micro_input_threshold <= 1
    assert config.model.engine.num_iterations > 0
    assert 0 <= config.dynamics.inhibition.neuron_factor <= 1
    assert 0 <= config.dynamics.inhibition.micro_factor <= 1
    assert 0 < config.dynamics.decay.micro_factor <= 1
    assert 0 < config.dynamics.decay.neuron_factor <= 1
    assert config.activation.function in ["sigmoid", "relu", "tanh"]
    
    # Validaciones para campos existentes del modelo
    assert 0 < config.model.engine.macro_neuron_threshold <= 1
    assert 0 <= config.model.engine.macro_initial_transition <= 1
    assert 0 < config.model.neuron.initial_threshold <= 1
    assert 0 <= config.model.neuron.initial_decay <= 1
    assert 0 < config.model.neuron.min_weight <= config.model.neuron.max_weight <= 1

    # 🌟 NUEVAS VALIDACIONES para umbrales de aprendizaje
    assert 0 < config.model.neuron.learning_threshold <= 1
    assert 0 < config.model.neuron.micro_learning_threshold <= 1
    assert config.model.neuron.learning_threshold >= config.model.neuron.micro_learning_threshold, \
        "El umbral de aprendizaje de neuronas debe ser mayor o igual al de micros"

    # Validaciones para clustering
    assert 0 < config.clustering.cooccurrence_threshold <= 1
    assert config.clustering.context_window > 0
    assert config.clustering.detection_frequency > 0
    assert config.clustering.min_words_per_concept >= 2
    
    # Validaciones para paths
    assert isinstance(config.paths.clustering_state, str)
    assert isinstance(config.paths.fasttext_vectors, str)


# ===============================
# LOADER
# ===============================

def _load_config(path: str) -> AppConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    
    # Cargar configuración de embeddings
    embeddings_raw = raw.get("embeddings", {})
    embeddings_config = EmbeddingsConfig(
        dimension=embeddings_raw.get("dimension", 300),
        use_fasttext=embeddings_raw.get("use_fasttext", True),
        fallback_dimension=embeddings_raw.get("fallback_dimension", 64),
        normalize=embeddings_raw.get("normalize", True)
    )

    # Cargar configuración de engine con manejo de campos opcionales
    engine_raw = raw["model"]["engine"]
    engine_config = EngineConfig(
        micro_input_threshold=engine_raw["micro_input_threshold"],
        num_iterations=engine_raw["num_iterations"],
        feedback_strength=engine_raw["feedback_strength"],
        meta_adjust=MetaAdjustConfig(**engine_raw["meta_adjust"]),
        # Campos opcionales
        macro_neuron_threshold=engine_raw.get("macro_neuron_threshold", 0.5),
        macro_initial_transition=engine_raw.get("macro_initial_transition", 0.1)
    )

    # Cargar configuración de neuronas con manejo de campos opcionales
    neuron_raw = raw["model"]["neuron"]
    neuron_config = NeuronConfig(
        use_attention=neuron_raw["use_attention"],
        learning_rate=neuron_raw["learning_rate"],
        lambda_decay=neuron_raw["lambda_decay"],
        # Campos existentes opcionales
        initial_threshold=neuron_raw.get("initial_threshold", 0.6),
        initial_decay=neuron_raw.get("initial_decay", 0.25),
        min_weight=neuron_raw.get("min_weight", 0.3),
        max_weight=neuron_raw.get("max_weight", 1.0),
        # 🌟 NUEVOS CAMPOS (opcionales)
        learning_threshold=neuron_raw.get("learning_threshold", 0.85),
        micro_learning_threshold=neuron_raw.get("micro_learning_threshold", 0.7)
    )

    # Cargar configuración de paths (opcional, con valores por defecto)
    paths_raw = raw.get("paths", {})
    paths_config = PathsConfig(
        clustering_state=paths_raw.get("clustering_state", "data/clustering_state.pkl"),
        fasttext_vectors=paths_raw.get("fasttext_vectors", "data/fasttext_vectors.vec")
    )

    # Cargar configuración de clustering (opcional)
    clustering_raw = raw.get("clustering", {})
    clustering_config = ClusteringConfig(
        cooccurrence_threshold=clustering_raw.get("cooccurrence_threshold", 0.7),
        context_window=clustering_raw.get("context_window", 10),
        detection_frequency=clustering_raw.get("detection_frequency", 20),
        min_words_per_concept=clustering_raw.get("min_words_per_concept", 2)
    )

    # Cargar configuración de aprendizaje (con campo nuevo opcional)
    learning_raw = raw.get("learning", {})
    learning_config = LearningConfig(
        scheduler=learning_raw.get("scheduler"),
        grad_clip=learning_raw.get("grad_clip"),
        hebbian_enabled=learning_raw.get("hebbian_enabled", True)
    )

    config = AppConfig(
        model=ModelConfig(
            seed=raw["model"]["seed"],
            engine=engine_config,
            thinking=ThinkingConfig(**raw["model"]["thinking"]),
            neuron=neuron_config,
        ),
        dynamics=DynamicsConfig(
            inhibition=InhibitionConfig(**raw["dynamics"]["inhibition"]),
            decay=DecayConfig(**raw["dynamics"]["decay"]),
        ),
        activation=ActivationConfig(**raw["activation"]),
        learning=learning_config,
        regularization=RegularizationConfig(**raw["regularization"]),
        logging=LoggingConfig(**raw["logging"]),
        paths=paths_config,
        clustering=clustering_config,
        embeddings=embeddings_config,
    )

    _validate_config(config)
    return config


# ===============================
# SINGLETON ACCESS
# ===============================

_config_instance: Optional[AppConfig] = None


def get_config(path: str = "config/aura_config.yaml") -> AppConfig:
    global _config_instance
    if _config_instance is None:
        _config_instance = _load_config(path)
    return _config_instance