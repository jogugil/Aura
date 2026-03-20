"""
cognitive_engine.py (versión PyTorch con estados agrupados)
Motor cognitivo principal que orquesta todas las capas neuronales.
"""

import os
import torch
import concurrent.futures
import logging
import atexit
from typing import List, Tuple, Dict, Any, Optional


# Importaciones de los nuevos estados
from ..states.micro_state import MicroNeuronState   
from ..states.neuron_state import NeuronState 
from ..states.macro_state import MacroNeuronState 
from ..states.interconnector_state import InterconnectorState
from .memory import Memory, register_in_memory
from .personality import Personality
from ..utils.vector_index import VectorIndex
from ..utils.neural_events import NeuralEvent, NeuralEventPublisher
from ..utils.priority_manager import PriorityManager
from .config import get_config
from ..learning.concept_clustering import ConceptGenerator
from ..states.micro_state import MicroNeuronState
from ..inference.micro_inference import activate as activate_micro, reset as reset_micro, apply_decay as decay_micro
from ..states.neuron_state import NeuronState
from ..inference.neuron_inference import evaluate_neurons, reset_neurons  ,  apply_neuron_decay, lateral_inhibition
from ..states.macro_state import MacroNeuronState
from ..inference.macro_inference import  evaluate_macro_neurons,  reset_macro_neurons
# Configurar logger
logger = logging.getLogger(__name__)


class CognitiveEngine:
    """
    Motor cognitivo principal que orquesta todas las capas neuronales.
    Versión adaptada a estados tensoriales, manteniendo toda la funcionalidad original.
    """

    def __init__(self, memory: Memory, personality: Personality, device: torch.device = torch.device('cpu')):
        """
        Inicializador de la clase CognitiveEngine.
        Carga la configuración global y la almacena como atributo de instancia.
        
        Args:
            memory: Instancia de Memory para recuperación asociativa
            personality: Instancia de Personality para influir en activaciones
            device: Dispositivo PyTorch (CPU/GPU)
        """
        self.memory = memory
        self.personality = personality
        self.device = device
             
        atexit.register(self._save_clustering_state)

        # Cargar configuración global
        self.config = get_config()
        
        # Registrar la inicialización con los parámetros que existen
        logger.debug("Inicializando CognitiveEngine con configuración:")
        logger.debug(f"  - micro_input_threshold: {self.config.model.engine.micro_input_threshold}")
        logger.debug(f"  - num_iterations: {self.config.model.engine.num_iterations}")
        logger.debug(f"  - feedback_strength: {self.config.model.engine.feedback_strength}")

        
        logger.debug(f"  - macro_neuron_threshold: {self.config.model.engine.macro_neuron_threshold}")
        logger.debug(f"  - macro_initial_transition: {self.config.model.engine.macro_initial_transition}")

        logger.debug(f"  - meta_adjust.window: {self.config.model.engine.meta_adjust.window}")
        logger.debug(f"  - meta_adjust.low_threshold: {self.config.model.engine.meta_adjust.low_threshold}")
        logger.debug(f"  - meta_adjust.high_threshold: {self.config.model.engine.meta_adjust.high_threshold}")
        logger.debug(f"  - meta_adjust.threshold_adjust: {self.config.model.engine.meta_adjust.threshold_adjust}")
        logger.debug(f"  - meta_adjust.decay_adjust: {self.config.model.engine.meta_adjust.decay_adjust}")

        logger.debug(f"  - use_attention: {self.config.model.neuron.use_attention}")
        logger.debug(f"  - learning_rate: {self.config.model.neuron.learning_rate}")
        logger.debug(f"  - lambda_decay: {self.config.model.neuron.lambda_decay}")

        
        logger.debug(f"  - initial_threshold: {self.config.model.neuron.initial_threshold}")
        logger.debug(f"  - initial_decay: {self.config.model.neuron.initial_decay}")
        logger.debug(f"  - min_weight: {self.config.model.neuron.min_weight}")
        logger.debug(f"  - max_weight: {self.config.model.neuron.max_weight}")

        logger.debug(f"  - inhibition.neuron_factor: {self.config.dynamics.inhibition.neuron_factor}")
        logger.debug(f"  - inhibition.micro_factor: {self.config.dynamics.inhibition.micro_factor}")
        logger.debug(f"  - decay.micro_factor: {self.config.dynamics.decay.micro_factor}")
        logger.debug(f"  - decay.neuron_factor: {self.config.dynamics.decay.neuron_factor}")
        logger.debug(f"  - activation.function: {self.config.activation.function}")

        # Estados de las poblaciones (se asignarán externamente)
        self.micro_state: Optional[MicroNeuronState] = None
        self.neuron_state: Optional[NeuronState] = None
        self.macro_state: Optional[MacroNeuronState] = None
        self.interconnector_state: Optional[InterconnectorState] = None

        # Matriz de transiciones entre macro-neuronas (conceptos)
        self.transitions = None  # se inicializará cuando se tenga macro_state
        
        # Componentes deliberativos (inicializados externamente una vez)
        self.synthesizer = None
        self.macro_tn = None
        self.response_builder = None

        # Generador de conceptos
        self.concept_generator = None
        self.interaction_counter = 0

        # Obtener ruta del clustering (con fallback por si no está definida)
        self.clustering_path = self.config.paths.clustering_state
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(self.clustering_path), exist_ok=True)
        embedding_dim = self.config.embeddings.dimension
 
        # Índice vectorial (para búsqueda rápida)
        self.vector_index = VectorIndex(dim=embedding_dim, device=device)

        # Gestión de eventos y prioridades (se mantienen igual)
        self.event_publisher = NeuralEventPublisher()
        self.priority_manager = PriorityManager()
        self.executor = concurrent.futures.ThreadPoolExecutor()

        # Vocabulario de palabras clave (lista de IDs)
        self.keyword_vocabulary: List[str] = []

        # Historial de ciclos (para meta‑ajuste)
        self.cycle_history: List[Dict] = []

        self.last_active_micros: List[str] = []
        self.last_active_neurons: List[str] = []

    # ------------------------------------------------------------------
    # Métodos para asignar los estados (llamados por el loader)
    # ------------------------------------------------------------------
    def set_micro_state(self, state: MicroNeuronState):
        """
        Asigna el estado de micro-neuronas y actualiza estructuras dependientes.
        """
        logger.debug("Asignando micro_state al CognitiveEngine")
        self.micro_state = state
        # Actualizar vocabulario de palabras clave
        self.keyword_vocabulary = []
        if state is not None:
            logger.debug(f"micro_state tiene {state.n} neuronas")
            
            # ACTUALIZAR LA DIMENSIÓN DEL VECTOR INDEX SI ES NECESARIO
            if state.embeddings.shape[1] != self.vector_index.dim:
                logger.debug(f"Actualizando VectorIndex de dimensión {self.vector_index.dim} a {state.embeddings.shape[1]}")
                self.vector_index = VectorIndex(dim=state.embeddings.shape[1], device=self.device)
            
            for i, neuron_type in enumerate(state.types):
                if neuron_type == 'keyword':
                    self.keyword_vocabulary.append(state.ids[i])
                    logger.debug(f"  - Palabra clave añadida: {state.ids[i]}")
            
            # Añadir todos los embeddings al índice vectorial
            logger.debug(f"Añadiendo {len(state.ids)} embeddings al índice vectorial")
            for i, mid in enumerate(state.ids):
                self.vector_index.add_vector(mid, state.embeddings[i].tolist(), state.metadata[i])
            logger.debug(f"Índice vectorial actualizado con {len(state.ids)} vectores")
    def set_neuron_state(self, state: NeuronState):
        """
        Asigna el estado de neuronas intermedias.
        
        Args:
            state: Estado de neuronas a asignar
        """
        logger.debug(f"Asignando neuron_state con {state.N if state else 0} neuronas")
        self.neuron_state = state

    def set_macro_state(self, state: MacroNeuronState):
        """
        Asigna el estado de macro-neuronas e inicializa matriz de transiciones.
        
        Args:
            state: Estado de macro-neuronas a asignar
        """
        logger.debug(f"Asignando macro_state con {state.Nm if state else 0} macro-neuronas")
        self.macro_state = state
        if state is not None and state.Nm > 0:
            # Obtener valor de transición inicial desde configuración
            initial_transition = self.config.model.engine.macro_initial_transition
            logger.debug(f"Inicializando matriz de transiciones con valor {initial_transition}")
            self.transitions = torch.full((state.Nm, state.Nm), initial_transition, device=self.device)
            logger.debug(f"Matriz de transiciones creada con forma: {self.transitions.shape}")

    def set_interconnector_state(self, state: InterconnectorState):
        """
        Asigna el estado de neuronas interconectoras.
        
        Args:
            state: Estado de interconectoras a asignar
        """
        logger.debug(f"Asignando interconnector_state con {len(state.ids) if state else 0} interconectoras")
        self.interconnector_state = state

    # ------------------------------------------------------------------
    # Métodos originales (adaptados)
    # ------------------------------------------------------------------
    def register_interconnector(self, interconnector):  # solo para compatibilidad
        """
        Método de compatibilidad para versiones anteriores.
        
        Args:
            interconnector: Objeto interconectora a registrar
        """
        logger.debug("Método register_interconnector llamado (compatibilidad)")
        # En la nueva versión, las interconectoras se cargan todas a la vez.
        pass


    def register_micro_neuron(self, mn):  # solo compatibilidad
        """
        Método de compatibilidad para versiones anteriores.
        
        Args:
            mn: Micro-neurona a registrar
        """
        logger.debug("Método register_micro_neuron llamado (compatibilidad)")
        pass

    def register_neuron(self, n):  # compatibilidad
        """
        Método de compatibilidad para versiones anteriores.
        
        Args:
            n: Neurona a registrar
        """
        logger.debug("Método register_neuron llamado (compatibilidad)")
        pass

    def register_macro_neuron(self, macro_n):  # compatibilidad
        """
        Método de compatibilidad para versiones anteriores.
        
        Args:
            macro_n: Macro-neurona a registrar
        """
        logger.debug("Método register_macro_neuron llamado (compatibilidad)")
        pass

    def reset(self):
        """Resetea todas las activaciones de todas las capas neuronales."""
        logger.debug("Reseteando todas las capas neuronales")
        
        if self.micro_state:
            logger.debug(f"Reseteando micro_state con {self.micro_state.n} neuronas")
            self.micro_state = reset_micro(self.micro_state)
            logger.debug("  - micro_state reseteado")
            
        if self.neuron_state:
            logger.debug(f"Reseteando neuron_state con {self.neuron_state.N} neuronas")
            self.neuron_state = reset_neurons(self.neuron_state)
            logger.debug("  - neuron_state reseteado")
            
        if self.macro_state and self.macro_state.Nm > 0:
            logger.debug(f"Reseteando macro_state con {self.macro_state.Nm} macro-neuronas")
            self.macro_state = reset_macro_neurons(self.macro_state)
            logger.debug("  - macro_state reseteado")

    def evaluate_neuron_layer(self, micro_activations: torch.Tensor) -> torch.Tensor:
        """
        Evalúa la capa de neuronas dadas las activaciones de micro‑neuronas.
        Retorna un tensor booleano con las neuronas activas.
        
        Args:
            micro_activations: Tensor con niveles de activación de micro-neuronas
            
        Returns:
            Tensor booleano con neuronas activas
        """
        if self.neuron_state is None:
            logger.debug("neuron_state es None, retornando tensor vacío")
            return torch.tensor([], device=self.device)
            
        logger.debug(f"Evaluando capa de neuronas con {len(micro_activations)} micro-activaciones")
        
        # Obtener parámetros de configuración
        use_attention = self.config.model.neuron.use_attention
        learning_rate = self.config.model.neuron.learning_rate
        lambda_decay = self.config.model.neuron.lambda_decay
        activation_function = self.config.activation.function
        
        # Seleccionar función de activación según configuración
        if activation_function == "sigmoid":
            activation_fn = torch.sigmoid
        elif activation_function == "relu":
            activation_fn = torch.relu
        elif activation_function == "tanh":
            activation_fn = torch.tanh
        else:
            logger.warning(f"Función de activación {activation_function} no reconocida, usando sigmoid")
            activation_fn = torch.sigmoid
            
        learning_threshold = self.config.model.neuron.learning_threshold
        micro_learning_threshold = self.config.model.neuron.micro_learning_threshold
    
        logger.debug(f"  - learning_threshold: {learning_threshold}")
        logger.debug(f"  - micro_learning_threshold: {micro_learning_threshold}")
        logger.debug(f"  - use_attention: {use_attention}")
        logger.debug(f"  - learning_rate: {learning_rate}")
        logger.debug(f"  - lambda_decay: {lambda_decay}")
        logger.debug(f"  - activation_function: {activation_function}")
        
        self.neuron_state, active = evaluate_neurons(
            self.neuron_state,
            micro_activations,
            activation_fn=activation_fn,
            use_attention=use_attention,
            learning_rate=learning_rate,
            lambda_decay=lambda_decay,
            learning_threshold=learning_threshold,               
            micro_learning_threshold=micro_learning_threshold    
        )
        
        num_active = active.sum().item()
        logger.debug(f"Neuronas activas después de evaluación: {num_active}")
        
        return active

    def _apply_neuron_to_micro_feedback(self, feedback_strength: float = None):
        """
        Aplica feedback desde neuronas activas a sus micro‑neuronas de entrada.
        
        Args:
            feedback_strength: Fuerza del feedback (si None, usa valor de configuración)
        """
        if self.neuron_state is None or self.micro_state is None:
            logger.debug("No se puede aplicar feedback: neuron_state o micro_state es None")
            return
            
        # Usar feedback_strength de configuración si no se proporciona
        if feedback_strength is None:
            feedback_strength = self.config.model.engine.feedback_strength
            
        logger.debug(f"Aplicando feedback con fuerza {feedback_strength}")
        
        # Matriz de pesos (Nn, Nm) dispersa
        feedback = torch.sparse.mm(self.neuron_state.weights.t(), 
                                   self.neuron_state.activation_level.unsqueeze(1)).squeeze(1)
        
        logger.debug(f"Feedback calculado: min={feedback.min().item():.4f}, "
                    f"max={feedback.max().item():.4f}, mean={feedback.mean().item():.4f}")
        
        self.micro_state.activation_level += feedback * feedback_strength
        self.micro_state.activation_level = torch.clamp(self.micro_state.activation_level, 0.0, 1.0)
        self.micro_state.active = self.micro_state.activation_level >= self.micro_state.activation_threshold
        
        logger.debug(f"Micro activas después de feedback: {self.micro_state.active.sum().item()}")

    def _retrieve_memory_from_activation(self, active_dict: Dict[str, bool]) -> Dict:
        """
        Recupera memoria asociativa basada en IDs de neuronas activas.
        
        Args:
            active_dict: Diccionario {id: bool} de neuronas activas
            
        Returns:
            Diccionario con memorias recuperadas
        """
        active_concepts = [nid for nid, act in active_dict.items() if act]
        
        if not active_concepts:
            logger.debug("No hay conceptos activos para recuperar memoria")
            return {}
            
        logger.debug(f"Recuperando memoria para {len(active_concepts)} conceptos activos")
        logger.debug(f"  Conceptos: {active_concepts[:10]}")
        
        activation_levels = [1.0] * len(active_concepts)
        memories = self.memory.retrieve_associative(active_concepts, activation_levels)
        
        logger.debug(f"Memorias recuperadas: {len(memories)}")
        
        return memories

    def _incorporate_retrieved_memories(self, retrieved_memories: Dict, strength: float = 0.1):
        """
        Refuerza activaciones de neuronas y micro‑neuronas según memorias recuperadas.
        Busca en metadata la clave 'memory_concept_id' para asociar.
        
        Args:
            retrieved_memories: Diccionario de memorias recuperadas
            strength: Fuerza de incorporación de memorias
        """
        if not retrieved_memories:
            logger.debug("No hay memorias para incorporar")
            return
            
        if self.micro_state is None or self.neuron_state is None:
            logger.debug("No se pueden incorporar memorias: micro_state o neuron_state es None")
            return

        logger.debug(f"Incorporando {len(retrieved_memories)} memorias con strength={strength}")

        # Crear mapeo de ID a índice para micro y neuronas
        micro_id_to_idx = {mid: i for i, mid in enumerate(self.micro_state.ids)}
        neuron_id_to_idx = {nid: i for i, nid in enumerate(self.neuron_state.ids)}
        
        incorporations = 0

        for concept_id, info in retrieved_memories.items():
            act = info.get('activation', 1.0) * strength * 0.5

            # Buscar en micro‑neuronas
            if concept_id in micro_id_to_idx:
                idx = micro_id_to_idx[concept_id]
                before = self.micro_state.activation_level[idx].item()
                self.micro_state.activation_level[idx] = min(1.0, self.micro_state.activation_level[idx] + act)
                after = self.micro_state.activation_level[idx].item()
                self.micro_state.active[idx] = self.micro_state.activation_level[idx] >= self.micro_state.activation_threshold[idx]
                logger.debug(f"  - Micro {concept_id}: activación {before:.3f} -> {after:.3f}")
                incorporations += 1

            # Buscar en neuronas
            if concept_id in neuron_id_to_idx:
                idx = neuron_id_to_idx[concept_id]
                before = self.neuron_state.activation_level[idx].item()
                self.neuron_state.activation_level[idx] = min(1.0, self.neuron_state.activation_level[idx] + act)
                after = self.neuron_state.activation_level[idx].item()
                self.neuron_state.active[idx] = self.neuron_state.activation_level[idx] >= self.neuron_state.activation_threshold[idx]
                logger.debug(f"  - Neurona {concept_id}: activación {before:.3f} -> {after:.3f}")
                incorporations += 1
                
        logger.debug(f"Total incorporaciones realizadas: {incorporations}")

    def _apply_decay(self):
        """Aplica decaimiento a todas las capas neuronales usando factores de configuración."""
        logger.debug("Aplicando decaimiento a todas las capas")
        
        micro_factor = self.config.dynamics.decay.micro_factor
        neuron_factor = self.config.dynamics.decay.neuron_factor
        
        logger.debug(f"  - micro_factor: {micro_factor}")
        logger.debug(f"  - neuron_factor: {neuron_factor}")

        if self.micro_state:
            before_micro = self.micro_state.active.sum().item()
            self.micro_state.activation_level *= micro_factor
            self.micro_state.active = (
                self.micro_state.activation_level >= self.micro_state.activation_threshold
            )
            after_micro = self.micro_state.active.sum().item()
            logger.debug(f"  - Micro: {before_micro} -> {after_micro} activas")

        if self.neuron_state:
            before_neuron = self.neuron_state.active.sum().item()
            self.neuron_state.activation_level *= neuron_factor
            self.neuron_state.active = (
                self.neuron_state.activation_level >= self.neuron_state.activation_threshold
            )
            after_neuron = self.neuron_state.active.sum().item()
            logger.debug(f"  - Neuronas: {before_neuron} -> {after_neuron} activas")
    
    def _save_clustering_state(self):
        """Guarda el estado actual del generador de conceptos."""
        if self.concept_generator and hasattr(self, 'clustering_path'):
            try:
                self.concept_generator.save_state(self.clustering_path)
                logger.debug(f"Estado de clustering guardado en {self.clustering_path}")
            except Exception as e:
                logger.error(f"Error al guardar estado de clustering: {e}")

    def initialize_deliberation(self, personality, language="es"):
        """
        Inicializa los módulos de deliberación (TN, MacroTN, Synthesizer, ResponseBuilder)
        y los inyecta en el motor cognitivo.
        
        Esta función debe llamarse después de cargar las neuronas y antes del bucle
        conversacional si se desea que el sistema tenga capacidad de deliberación
        y generación de respuestas.
        
        Args:
            personality: Instancia de Personality con la configuración de la IA
            language: Código de idioma para las respuestas ('es' o 'en')
        
        Returns:
            None (los módulos quedan almacenados en el motor)
        """
        logger.info("=" * 50)
        logger.info("Inicializando módulos de deliberación")
        logger.debug(f"Idioma seleccionado: {language}")
        
        # ------------------------------------------------------------------
        # 1. IMPORTACIONES LOCALES (evitan dependencias circulares)
        # ------------------------------------------------------------------
        from ..deliberation.thinking_neurons import populate_tns
        from ..language.grammar_adjudicator import GrammarAdjudicator
        from ..deliberation.context_synthesizer import ContextSynthesizer
        from ..deliberation.response_builder import ResponseBuilder
        
        logger.debug("Importaciones realizadas correctamente")
        
        # ------------------------------------------------------------------
        # 2. POBLAR THINKING NEURONS Y MACROTN
        # ------------------------------------------------------------------
        # Las Thinking Neurons son agentes especializados que proponen planes
        # desde diferentes perspectivas (social, lógica, ambigüedad, etc.)
        # MacroTN es el coordinador que selecciona el mejor plan
        logger.info("Poblando Thinking Neurons y MacroTN...")
        
        try:
            tn_list, macro_tn = populate_tns(
                memory=self.memory,
                neural_system=self,
                interconnectors=self.interconnector_state,
                engine=self
            )
            self.macro_tn = macro_tn
            logger.info(f"✅ Thinking Neurons pobladas: {len(tn_list)} especializadas")
            logger.debug(f"Tipos de TN: {[tn.__class__.__name__ for tn in tn_list]}")
        except Exception as e:
            logger.error(f"Error al poblar Thinking Neurons: {e}", exc_info=True)
            raise
      
        # ------------------------------------------------------------------
        # 3. CREAR SINTETIZADOR DE CONTEXTO
        # ------------------------------------------------------------------
        # El sintetizador genera hipótesis de contexto a partir de las
        # activaciones neuronales y las memorias recuperadas
        logger.info("Creando Sintetizador de Contexto...")
      
        try:
            self.synthesizer = ContextSynthesizer(self)
            logger.debug("Sintetizador creado correctamente")
        except Exception as e:
            logger.error(f"Error al crear Sintetizador: {e}", exc_info=True)
            raise
      
        # ------------------------------------------------------------------
        # 4. CREAR ADJUDICADOR GRAMATICAL
        # ------------------------------------------------------------------
        # El adjudicador gestiona las reglas gramaticales y las transiciones
        # entre conceptos para la generación de respuestas
        logger.info("Creando Grammar Adjudicator...")
      
        try:
            adjudicator = GrammarAdjudicator()
            logger.debug("Adjudicador gramatical creado correctamente")
        except Exception as e:
            logger.error(f"Error al crear GrammarAdjudicator: {e}", exc_info=True)
            raise
      
        # ------------------------------------------------------------------
        # 5. CREAR RESPONSE BUILDER (CAPA DE EXPRESIÓN)
        # ------------------------------------------------------------------
        # El response builder transforma los planes conceptuales en respuestas
        # en lenguaje natural, usando el mapeo concepto → palabras
        logger.info("Creando ResponseBuilder...")
      
        try:
            self.response_builder = ResponseBuilder(
                personality=personality,
                context=None,
                winning_plan=None,
                adjudicator=adjudicator,
                engine=self,
                interconnectors=self.interconnector_state
            )
            self.response_builder.language = language
            logger.debug(f"ResponseBuilder creado con idioma: {language}")
          
            # Verificar que el mapeo concepto→palabras se ha construido
            num_concepts = len(self.response_builder.concept_to_words)
            logger.debug(f"Concept_to_words contiene {num_concepts} conceptos")
          
        except Exception as e:
            logger.error(f"Error al crear ResponseBuilder: {e}", exc_info=True)
            raise
          
        # ------------------------------------------------------------------
        # 6. INICIALIZAR GENERADOR DE CONCEPTOS (CLUSTERING)
        # ------------------------------------------------------------------
        threshold = self.config.clustering.cooccurrence_threshold
        window = self.config.clustering.context_window
        self.concept_generator = ConceptGenerator(
            micro_state=self.micro_state,
            macro_state=self.macro_state,
            cooccurrence_threshold=threshold,
            window=window
        )
     
        logger.info("✅ Generador de conceptos inicializado")
        
        # ------------------------------------------------------------------
        # 7. VERIFICACIÓN FINAL
        # ------------------------------------------------------------------
        if all([self.macro_tn, self.synthesizer, self.response_builder]):
            logger.info("✅ Módulos de deliberación inicializados correctamente")
            logger.info(f"   - MacroTN: {type(self.macro_tn).__name__}")
            logger.info(f"   - Synthesizer: {type(self.synthesizer).__name__}")
            logger.info(f"   - ResponseBuilder: {type(self.response_builder).__name__}")
            logger.info(f"   - Idioma: {language}")
        else:
            logger.error("❌ Algunos módulos no se inicializaron correctamente")
            logger.debug(f"macro_tn: {self.macro_tn is not None}")
            logger.debug(f"synthesizer: {self.synthesizer is not None}")
            logger.debug(f"response_builder: {self.response_builder is not None}")
      
        logger.info("=" * 50)

    def iterative_process_input(self,
                               input_vectors: torch.Tensor,
                               original_phrase: Optional[str] = None,
                               micro_threshold: float = None,
                               num_iterations: int = None) -> Dict[str, Any]:
        """
        Procesa la entrada iterativamente con todas las capas.
        
        Args:
            input_vectors: tensor (T, dim) o (dim,)
            original_phrase: Frase original de entrada (opcional)
            micro_threshold: Umbral para micro-neuronas (si None, usa valor de configuración)
            num_iterations: Número de iteraciones (si None, usa valor de configuración)
            
        Returns:
            Diccionario con activaciones finales
        """
        logger.info("Iniciando procesamiento iterativo de entrada")
        
        # Usar valores de configuración si no se proporcionan
        if micro_threshold is None:
            micro_threshold = self.config.model.engine.micro_input_threshold
        if num_iterations is None:
            num_iterations = self.config.model.engine.num_iterations
            
        logger.debug(f"Parámetros: micro_threshold={micro_threshold}, num_iterations={num_iterations}")
        
        self.reset()
        self.memory.clear_memory("thinking")
        logger.debug("Reset completado y memoria thinking limpiada")

        if input_vectors.dim() == 1:
            input_vectors = input_vectors.unsqueeze(0)
        T, dim = input_vectors.shape
        logger.debug(f"Entrada: {T} vectores de dimensión {dim}")

        # --- Activación inicial de micro‑neuronas ---
        query_vec = input_vectors[0].cpu().tolist()
        similar = self.vector_index.search_similar(query_vec, top_k=10)
        logger.debug(f"Búsqueda de similitud: {len(similar)} resultados")
        if similar:
            logger.debug(f"Top resultados: {similar[:5]}")

        if self.micro_state is not None:
            logger.debug("Activando micro-neuronas con vector de entrada")
            
            # Obtener función de activación de configuración
            activation_function = self.config.activation.function
            if activation_function == "sigmoid":
                activation_fn = torch.sigmoid
            elif activation_function == "relu":
                activation_fn = torch.relu
            elif activation_function == "tanh":
                activation_fn = torch.tanh
            else:
                logger.warning(f"Función de activación {activation_function} no reconocida, usando sigmoid")
                activation_fn = torch.sigmoid
                
            self.micro_state, active_micro = activate_micro(
                self.micro_state,
                input_vectors[0],
                original_phrase=original_phrase,
                threshold=micro_threshold,
                activation_fn=activation_fn
            )
            activated_mn_ids = [self.micro_state.ids[i] for i in range(self.micro_state.n) if active_micro[i]]
            initial_activations = {mid: self.micro_state.activation_level[self.micro_state.ids.index(mid)].item()
                                  for mid in activated_mn_ids}
            logger.debug(f"Micros activas tras activación inicial: {len(activated_mn_ids)}")
            if activated_mn_ids:
                logger.debug(f"IDs micro activas (primeras 10): {activated_mn_ids[:10]}")
                for mid in activated_mn_ids[:5]:
                    idx = self.micro_state.ids.index(mid)
                    logger.debug(f"   {mid}: activation={self.micro_state.activation_level[idx].item():.3f}, "
                              f"threshold={self.micro_state.activation_threshold[idx].item():.3f}")
        else:
            activated_mn_ids = []
            initial_activations = {}
            logger.debug("micro_state es None")

        # Ajuste por interconectoras (omitido)
        if self.interconnector_state is not None and self.micro_state is not None:
            logger.debug("Aplicando ajuste por interconectoras (pendiente implementación)")
            pass

        # Publicar eventos y priority manager
        for mn_id in activated_mn_ids:
            idx = self.micro_state.ids.index(mn_id)
            self.event_publisher.publish(NeuralEvent("neuron_activated", {
                "neuron_id": mn_id,
                "neuron_type": "micro",
                "activation_level": self.micro_state.activation_level[idx].item()
            }))
            self.priority_manager.add_item(mn_id, priority=1)
        logger.debug(f"Eventos publicados para {len(activated_mn_ids)} micro-neuronas")

        # Registrar en thinking memory
        # Umbral dinámico basado en activaciones reales
        if initial_activations:
            max_act = max(initial_activations.values())
            config_threshold = self.config.model.thinking.thinking_memory_threshold
            thinking_threshold = min(config_threshold, max_act * 0.85)
        else:
            thinking_threshold = self.config.model.thinking.thinking_memory_threshold

        logger.debug(f"Umbral thinking dinámico: {thinking_threshold:.3f}")
        thinking_records = 0
        for mn_id in activated_mn_ids:
            idx = self.micro_state.ids.index(mn_id)
            if initial_activations.get(mn_id, 0) >= thinking_threshold and self.micro_state.types[idx] == 'keyword':
                key_info = {
                    "id": mn_id,
                    "type": "keyword",
                    "initial_activation": initial_activations[mn_id],
                    "metadata": self.micro_state.metadata[idx]
                }
                self.memory.add_to_memory(key_info, "thinking")
                thinking_records += 1
        logger.debug(f"Registros añadidos a thinking memory: {thinking_records}")

        while not self.priority_manager.is_empty():
            self.priority_manager.get_next_item()

        # --- Bucle iterativo ---
        logger.info(f"Iniciando bucle iterativo de {num_iterations} iteraciones")
        
        for it in range(num_iterations):
            logger.debug(f"Iteración {it+1}/{num_iterations}")
            logger.debug(f"  Estado inicial: micros activas = {self.micro_state.active.sum().item()}, "
                      f"neuronas activas = {self.neuron_state.active.sum().item() if self.neuron_state else 0}")

            # 1. Propagación hacia adelante (micro -> neuronas)
            logger.debug("  Paso 1/8: Propagación micro -> neuronas")
            active_neurons = self.evaluate_neuron_layer(self.micro_state.activation_level)
            if self.neuron_state:
                logger.debug(f"    Neuronas activas después de evaluación: {self.neuron_state.active.sum().item()}")
                # Mostrar las primeras neuronas activas
                active_ids = [self.neuron_state.ids[i] for i in range(self.neuron_state.N) if self.neuron_state.active[i]]
                if active_ids:
                    logger.debug(f"    IDs activas (primeras 5): {active_ids[:5]}")

            # 2. Aprendizaje Hebbiano (ya dentro de evaluate_neurons)

            # 3. Inhibición lateral
            if self.neuron_state:
                logger.debug("  Paso 2/8: Aplicando inhibición lateral")
                before_inhib = self.neuron_state.active.sum().item()
                self.neuron_state = lateral_inhibition(self.neuron_state)
                after_inhib = self.neuron_state.active.sum().item()
                logger.debug(f"    Neuronas activas después de inhibición: {before_inhib} -> {after_inhib}")

            # 4. Evaluar macro‑neuronas
            logger.debug("  Paso 3/8: Evaluando macro-neuronas")
            if self.macro_state is not None and self.macro_state.Nm > 0:
                if self.neuron_state is not None and self.micro_state is not None:
                    active_neuron_indices = self.neuron_state.active.nonzero(as_tuple=True)[0]
                    active_micro_indices = self.micro_state.active.nonzero(as_tuple=True)[0]
                    logger.debug(f"    Antes de macro: neuronas activas idx = {active_neuron_indices.tolist()}, "
                              f"micros activas idx = {active_micro_indices.tolist()}")
                    
                    self.macro_state, macro_active = evaluate_macro_neurons(
                        self.macro_state,
                        active_neuron_indices,
                        active_micro_indices
                    )
                    logger.debug(f"    Macro activas = {macro_active.sum().item()}")
                    
                    if macro_active.sum() > 0:
                        active_macro_ids = [self.macro_state.ids[i] for i in range(self.macro_state.Nm) if macro_active[i]]
                        logger.debug(f"    IDs macro activas: {active_macro_ids}")

                    # Inhibición por macro
                    logger.debug("    Aplicando inhibición por macro-neuronas")
                    active_macro_indices_local = macro_active.nonzero(as_tuple=True)[0]
                    
                    # Obtener factores de inhibición de configuración
                    neuron_inhibition_factor = self.config.dynamics.inhibition.neuron_factor
                    micro_inhibition_factor = self.config.dynamics.inhibition.micro_factor
                    
                    logger.debug(f"    Factores inhibición: neuronas={neuron_inhibition_factor}, micros={micro_inhibition_factor}")
                    
                    for idx in active_macro_indices_local:
                        cond_n_idx = self.macro_state.condition_indices[idx]
                        excl_mn_idx = self.macro_state.exclusion_indices[idx]
                        
                        if self.neuron_state is not None:
                            mask_n = torch.ones(self.neuron_state.N, dtype=torch.bool, device=self.device)
                            if cond_n_idx:
                                mask_n[cond_n_idx] = False
                            self.neuron_state.activation_level[mask_n] *= neuron_inhibition_factor
                            logger.debug(f"      Inhibición neuronas: {len(cond_n_idx)} excluidas, factor={neuron_inhibition_factor}")
                            
                        if self.micro_state is not None:
                            mask_m = torch.ones(self.micro_state.n, dtype=torch.bool, device=self.device)
                            if excl_mn_idx:
                                mask_m[excl_mn_idx] = False
                            self.micro_state.activation_level[mask_m] *= micro_inhibition_factor
                            logger.debug(f"      Inhibición micros: {len(excl_mn_idx)} excluidas, factor={micro_inhibition_factor}")
                            
                    self.neuron_state.active = self.neuron_state.activation_level >= self.neuron_state.activation_threshold
                    self.micro_state.active = self.micro_state.activation_level >= self.micro_state.activation_threshold
            else:
                logger.debug("    No hay macro-neuronas o macro_state es None")

            # 5. Feedback de neuronas a micro
            logger.debug("  Paso 4/8: Aplicando feedback neuronas -> micro")
            self._apply_neuron_to_micro_feedback()
            logger.debug(f"    Micros activas después de feedback: {self.micro_state.active.sum().item()}")

            # 6. Recuperación de memoria asociativa
            logger.debug("  Paso 5/8: Recuperando memoria asociativa")
            active_dict = {}
            if self.neuron_state:
                for i, act in enumerate(self.neuron_state.active):
                    if act:
                        active_dict[self.neuron_state.ids[i]] = True
            memories = self._retrieve_memory_from_activation(active_dict)
            logger.debug(f"    Memorias recuperadas: {len(memories)}")
            if memories:
                logger.debug(f"    {list(memories.keys())[:5]}")
            self._incorporate_retrieved_memories(memories)

            # 7. Decaimiento
            logger.debug("  Paso 6/8: Aplicando decaimiento")
            self._apply_decay()
            logger.debug(f"    Después decaimiento: micros activas = {self.micro_state.active.sum().item()}, "
                      f"neuronas activas = {self.neuron_state.active.sum().item()}")

            # 8. Guardar historial
            logger.debug("  Paso 7/8: Guardando historial del ciclo")
            cycle_info = {
                'iteration': it,
                'micro_activations': {mid: {'activation_level': self.micro_state.activation_level[i].item()}
                                      for i, mid in enumerate(self.micro_state.ids)},
                'neuron_details': {nid: {'activation_level': self.neuron_state.activation_level[i].item()}
                                  for i, nid in enumerate(self.neuron_state.ids)},
                'macro_active': macro_active.tolist() if 'macro_active' in locals() else []
            }
            self.cycle_history.append(cycle_info)

            logger.debug(f"  Paso 8/8: Fin de iteración {it+1}")

        # Estado final
        logger.info("Procesamiento iterativo completado, generando estado final")
        
        final_state = {
            'micro_neurons': {mid: self.micro_state.active[i].item()
                             for i, mid in enumerate(self.micro_state.ids)},
            'neurons': {nid: self.neuron_state.active[i].item()
                       for i, nid in enumerate(self.neuron_state.ids)}
        }
        
        self.last_active_micros = [mid for mid, act in final_state['micro_neurons'].items() if act]


        # ============================================================
        # CLUSTERING: APRENDIZAJE DE NUEVOS CONCEPTOS
        # ============================================================
        if self.concept_generator:
            self.concept_generator.register_activations(self.last_active_micros)
            self.interaction_counter += 1

            # Cada 20 interacciones, buscar nuevos conceptos (ajusta la frecuencia)
            if self.interaction_counter % 20 == 0:
                new_concepts = self.concept_generator.detect_new_concepts()
                for group, confidence in new_concepts:
                    logger.info(f"🎉 Nuevo grupo detectado con confianza {confidence:.3f}: {group}")
                     
                    concept_id = self.concept_generator.create_new_concept(group, confidence, self)
                    if concept_id:
                        self._save_clustering_state()  # Guardar inmediatamente
                        self.last_active_neurons = [nid for nid, act in final_state['neurons'].items() if act]

                logger.info(f"Estado final: micros activas = {len(self.last_active_micros)}, "
                          f"neuronas activas = {len(self.last_active_neurons)}")
                
                logger.debug(f"IDs micros activas: {self.last_active_micros[:20]}")
                logger.debug(f"IDs neuronas activas: {self.last_active_neurons[:20]}")
        
        # ============================================================
        # FASE DELIBERATIVA (si está inicializada)
        # ============================================================

        response = None
        winning_plan = None

        if self.synthesizer and self.macro_tn and self.response_builder:
            logger.info("Iniciando fase deliberativa")

            neural_state = {
                'micro_neurons': final_state['micro_neurons'],
                'neurons': final_state['neurons']
            }

            retrieved = {}

            logger.debug("Generando hipótesis de contexto")
            context_hypotheses = self.synthesizer.synthesize(
                neural_state,
                retrieved
            )
            logger.debug(f"Hipótesis generadas: {len(context_hypotheses)}")
            logger.debug(f"Hipótesis detalle: {context_hypotheses}")

            logger.debug("Ejecutando MacroTN")
            winning_plan, _, _ = self.macro_tn.reasoning_cycle(
                neural_state,
                retrieved,
                context_hypotheses
            )

            if winning_plan:
                logger.debug(f"Plan ganador: {winning_plan.get('conceptual_plan')}")
                self.response_builder.winning_plan = winning_plan
                response = self.response_builder.build_response()
                logger.info(f"Respuesta generada: {response}")
            else:
                logger.warning("No se generó plan ganador")
                response = "No estoy seguro de cómo responder."
        else:
            logger.debug("Módulos deliberativos no inicializados")

        return {
            "neural_state": final_state,
            "response": response,
            "plan": winning_plan
        }
        
    def reinforce_proposal(self,
                          successful_proposal: bool,
                          active_micro_ids: List[str],
                          active_neuron_ids: List[str],
                          success_rate: float = None,
                          failure_rate: float = None):
        """
        Refuerza o debilita las conexiones entre las micro‑neuronas y neuronas
        que participaron en la generación de la respuesta, según si fue exitosa.
        
        Args:
            successful_proposal: True si la propuesta fue exitosa, False si no
            active_micro_ids: IDs de micro-neuronas activas
            active_neuron_ids: IDs de neuronas activas
            success_rate: Tasa de refuerzo para éxito (si None, usa valor por defecto)
            failure_rate: Tasa de refuerzo para fracaso (si None, usa valor por defecto)
        """
        logger.info(f"Reforzando propuesta: {'éxito' if successful_proposal else 'fracaso'}")
        
        if self.neuron_state is None or self.micro_state is None:
            logger.error("No se puede reforzar: neuron_state o micro_state no inicializado")
            return

        # Usar valores por defecto si no se proporcionan
        if success_rate is None:
            success_rate = 0.05
        if failure_rate is None:
            failure_rate = -0.02

        rate = success_rate if successful_proposal else failure_rate
        logger.debug(f"Tasa aplicada: {rate}")

        # Obtener límites de pesos de configuración
        min_weight = self.config.model.neuron.min_weight
        max_weight = self.config.model.neuron.max_weight

        # Para actualizar las existentes, necesitamos los índices y valores actuales
        current_indices = self.neuron_state.weights._indices()
        current_values = self.neuron_state.weights._values()
        
        logger.debug(f"Matriz pesos actual: {len(current_values)} conexiones no-cero")

        # Para cada neurona activa
        connections_updated = 0
        for n_id in active_neuron_ids:
            if n_id not in self.neuron_state.ids:
                logger.debug(f"  Neurona {n_id} no encontrada en neuron_state, ignorada")
                continue
                
            idx_n = self.neuron_state.ids.index(n_id)
            conditions = self.neuron_state.condition_indices[idx_n]
            logger.debug(f"  Neurona {n_id} tiene {len(conditions)} condiciones")

            for idx_m in conditions:
                mn_id = self.micro_state.ids[idx_m]
                if mn_id in active_micro_ids:
                    # Buscar si la conexión (idx_n, idx_m) existe
                    mask = (current_indices[0] == idx_n) & (current_indices[1] == idx_m)
                    if mask.any():
                        # Actualizar el valor existente
                        pos = mask.nonzero(as_tuple=True)[0][0]
                        previous_value = current_values[pos].item()
                        current_values[pos] += rate
                        current_values[pos] = torch.clamp(current_values[pos], min_weight, max_weight)
                        logger.debug(f"    Conexión {n_id} -> {mn_id}: weight {previous_value:.3f} -> {current_values[pos].item():.3f}")
                        connections_updated += 1
                    # Si no existe, podríamos crear una nueva conexión (opcional)

        logger.debug(f"Conexiones actualizadas: {connections_updated}")

        # Reconstruir matriz con los nuevos valores
        self.neuron_state.weights = torch.sparse_coo_tensor(
            current_indices, current_values,
            self.neuron_state.weights.shape,
            device=self.device
        )
        
        logger.debug("Matriz de pesos reconstruida")


    def add_macro_neuron(self, new_id: str, new_name: str,
                        condition_neuron_indices: List[int],  # índices de neuronas intermedias
                        exclusion_micro_indices: List[int],   # índices de micro‑neuronas
                        initial_threshold: float = None,
                        metadata: dict = None) -> bool:
        """
        Añade una nueva macro‑neurona al macro_state en caliente.
        
        Args:
            new_id: ID único de la macro-neurona
            new_name: Nombre descriptivo
            condition_neuron_indices: Índices de neuronas que activan esta macro
            exclusion_micro_indices: Índices de micro-neuronas que inhiben esta macro
            initial_threshold: Umbral de activación (si None, usa configuración)
            metadata: Metadatos adicionales
            
        Returns:
            bool: True si se añadió correctamente
        """
        if self.macro_state is None:
            logger.error("Error: macro_state no inicializado")
            return False

        ms = self.macro_state
        device = ms.device
        dtype = ms.dtype

        logger.debug(f"Añadiendo macro-neurona: {new_id}")

        # 1. Expandir listas Python
        ms.ids.append(new_id)
        ms.names.append(new_name)
        ms.metadata.append(metadata or {})
        ms.condition_indices.append(condition_neuron_indices)
        ms.exclusion_indices.append(exclusion_micro_indices)

        # 2. Actualizar matriz de condiciones (dispersa) – relaciones con neuronas
        current_cond_indices = ms.conditions._indices()
        current_cond_values = ms.conditions._values()
        new_rows = [ms.Nm] * len(condition_neuron_indices)
        new_cols = condition_neuron_indices
        if new_rows:
            new_cond_indices = torch.tensor([new_rows, new_cols], device=device)
            new_cond_values = torch.ones(len(condition_neuron_indices), device=device, dtype=dtype)
            current_cond_indices = torch.cat([current_cond_indices, new_cond_indices], dim=1)
            current_cond_values = torch.cat([current_cond_values, new_cond_values])
        new_Nm = ms.Nm + 1
        ms.conditions = torch.sparse_coo_tensor(
            current_cond_indices, current_cond_values, (new_Nm, ms.Nn), device=device
        )

        # 3. Actualizar matriz de exclusiones (dispersa) – relaciones con micro
        current_excl_indices = ms.exclusions._indices()
        current_excl_values = ms.exclusions._values()
        new_excl_rows = [ms.Nm] * len(exclusion_micro_indices)
        new_excl_cols = exclusion_micro_indices
        if new_excl_rows:
            new_excl_indices = torch.tensor([new_excl_rows, new_excl_cols], device=device)
            new_excl_values = torch.ones(len(exclusion_micro_indices), device=device, dtype=dtype)
            current_excl_indices = torch.cat([current_excl_indices, new_excl_indices], dim=1)
            current_excl_values = torch.cat([current_excl_values, new_excl_values])
        ms.exclusions = torch.sparse_coo_tensor(
            current_excl_indices, current_excl_values, (new_Nm, ms.Nmicro), device=device
        )

        # 4. condition_lengths
        ms.condition_lengths = torch.cat([
            ms.condition_lengths,
            torch.tensor([len(condition_neuron_indices)], device=device)
        ])

        # 5. umbral (usar configuración si no se proporciona)
        new_threshold = initial_threshold if initial_threshold is not None else self.config.model.engine.macro_neuron_threshold
        ms.threshold = torch.cat([
            ms.threshold,
            torch.tensor([new_threshold], device=device, dtype=dtype)
        ])

        # 6. active y activation_level
        ms.active = torch.cat([
            ms.active,
            torch.tensor([False], device=device)
        ])
        ms.activation_level = torch.cat([
            ms.activation_level,
            torch.tensor([0.0], device=device, dtype=dtype)
        ])

        # 7. transitions (expandir matriz cuadrada) - usar valor de configuración
        initial_transition = self.config.model.engine.macro_initial_transition
        new_transitions = torch.full((new_Nm, new_Nm), initial_transition, device=device, dtype=dtype)
        new_transitions[:ms.Nm, :ms.Nm] = ms.transitions
        ms.transitions = new_transitions

        # 8. historial
        ms.activation_history.append([])

        # 9. Actualizar Nm
        ms.Nm = new_Nm

        logger.info(f"[CognitiveEngine] Nueva macro‑neurona añadida: {new_id}")
        return True

    def add_neuron(self, new_id: str, new_name: str,
                  condition_micro_ids: List[str],   # IDs de micro‑neuronas que activan esta neurona
                  exclusion_micro_ids: List[str] = None,
                  initial_threshold: float = None,
                  initial_decay: float = None,
                  metadata: dict = None) -> bool:
        """
        Añade una nueva neurona intermedia al neuron_state en caliente.
        
        Args:
            new_id: ID único de la neurona
            new_name: Nombre descriptivo
            condition_micro_ids: IDs de micro-neuronas que activan esta neurona
            exclusion_micro_ids: IDs de micro-neuronas que inhiben esta neurona
            initial_threshold: Umbral de activación (si None, usa configuración)
            initial_decay: Tasa de decaimiento (si None, usa configuración)
            metadata: Metadatos adicionales
            
        Returns:
            bool: True si se añadió correctamente
        """
        if self.neuron_state is None or self.micro_state is None:
            logger.error("Error: neuron_state o micro_state no inicializado")
            return False

        ns = self.neuron_state
        ms = self.micro_state
        device = ns.device
        dtype = ns.dtype

        logger.debug(f"Añadiendo neurona: {new_id}")

        # Convertir IDs de micro a índices
        cond_indices = [ms.ids.index(mid) for mid in condition_micro_ids if mid in ms.ids]
        excl_indices = []
        if exclusion_micro_ids:
            excl_indices = [ms.ids.index(mid) for mid in exclusion_micro_ids if mid in ms.ids]

        # 1. Expandir listas Python
        ns.ids.append(new_id)
        ns.names.append(new_name)
        ns.metadata.append(metadata or {})
        ns.condition_indices.append(cond_indices)

        # 2. Actualizar matriz de pesos (dispersa) – añadir fila para la nueva neurona
        current_weight_indices = ns.weights._indices()
        current_weight_values = ns.weights._values()
        new_rows = [ns.N] * len(cond_indices)
        new_cols = cond_indices
        if new_rows:
            new_indices = torch.tensor([new_rows, new_cols], device=device)
            # Pesos iniciales aleatorios usando configuración
            min_weight = self.config.model.neuron.min_weight
            max_weight = self.config.model.neuron.max_weight
            new_values = torch.empty(len(cond_indices), device=device, dtype=dtype).uniform_(min_weight, max_weight)
            current_weight_indices = torch.cat([current_weight_indices, new_indices], dim=1)
            current_weight_values = torch.cat([current_weight_values, new_values])
        new_N = ns.N + 1
        ns.weights = torch.sparse_coo_tensor(
            current_weight_indices, current_weight_values, (new_N, ns.M), device=device
        )

        # 3. Actualizar matriz de exclusiones (dispersa)
        current_excl_indices = ns.exclusions._indices()
        current_excl_values = ns.exclusions._values()
        if excl_indices:
            new_excl_rows = [ns.N] * len(excl_indices)
            new_excl_cols = excl_indices
            new_excl_indices = torch.tensor([new_excl_rows, new_excl_cols], device=device)
            new_excl_values = torch.ones(len(excl_indices), device=device, dtype=dtype)
            current_excl_indices = torch.cat([current_excl_indices, new_excl_indices], dim=1)
            current_excl_values = torch.cat([current_excl_values, new_excl_values])
        ns.exclusions = torch.sparse_coo_tensor(
            current_excl_indices, current_excl_values, (new_N, ns.M), device=device
        )

        # 4. Umbral de activación (usar configuración si no se proporciona)
        new_threshold = initial_threshold if initial_threshold is not None else self.config.model.neuron.initial_threshold
        ns.activation_threshold = torch.cat([
            ns.activation_threshold,
            torch.tensor([new_threshold], device=device, dtype=dtype)
        ])

        # 5. Decay rate (usar configuración si no se proporciona)
        new_decay = initial_decay if initial_decay is not None else self.config.model.neuron.initial_decay
        ns.decay_rate = torch.cat([
            ns.decay_rate,
            torch.tensor([new_decay], device=device, dtype=dtype)
        ])

        # 6. Estado variable
        ns.activation_level = torch.cat([
            ns.activation_level,
            torch.tensor([0.0], device=device, dtype=dtype)
        ])
        ns.active = torch.cat([
            ns.active,
            torch.tensor([False], device=device)
        ])

        # 7. Frecuencia micro (se mantiene igual, no necesita expandirse)
        # 8. Historial
        ns.activation_history.append([])

        # 9. Actualizar N
        ns.N = new_N

        logger.info(f"[CognitiveEngine] Nueva neurona intermedia añadida: {new_id}")
        return True
    
    # Función para añadir una macro-neurona (originalmente dentro de cognitive_engine)
    def add_macro_neuron(self, new_id: str, new_name: str,
                        condition_indices: List[int],      # índices de neuronas (nivel intermedio)
                        exclusion_indices: List[int],      # índices de micro‑neuronas
                        initial_threshold: float = None,
                        metadata: dict = None) -> bool:
        """
        Añade una nueva macro‑neurona al macro_state en caliente.
        Retorna True si se añadió correctamente.
        """
        if self.macro_state is None:
            logger.debug("Error: macro_state no inicializado")
            return False

        ms = self.macro_state
        device = ms.device
        dtype = ms.dtype

        # 1. Expandir listas Python
        ms.ids.append(new_id)
        ms.names.append(new_name)
        ms.metadata.append(metadata or {})
        ms.condition_indices.append(condition_indices)
        ms.exclusion_indices.append(exclusion_indices)

        # 2. Actualizar matriz de condiciones (dispersa)
        cond_indices = ms.conditions._indices()
        cond_values = ms.conditions._values()
        new_rows = [ms.Nm] * len(condition_indices)
        new_cols = condition_indices
        if new_rows:
            new_cond_indices = torch.tensor([new_rows, new_cols], device=device)
            new_cond_values = torch.ones(len(condition_indices), device=device, dtype=dtype)
            cond_indices = torch.cat([cond_indices, new_cond_indices], dim=1)
            cond_values = torch.cat([cond_values, new_cond_values])
        new_Nm = ms.Nm + 1
        ms.conditions = torch.sparse_coo_tensor(
            cond_indices, cond_values, (new_Nm, ms.Nn), device=device
        )

        # 3. Actualizar matriz de exclusiones (dispersa)
        excl_indices = ms.exclusions._indices()
        excl_values = ms.exclusions._values()
        new_excl_rows = [ms.Nm] * len(exclusion_indices)
        new_excl_cols = exclusion_indices
        if new_excl_rows:
            new_excl_indices = torch.tensor([new_excl_rows, new_excl_cols], device=device)
            new_excl_values = torch.ones(len(exclusion_indices), device=device, dtype=dtype)
            excl_indices = torch.cat([excl_indices, new_excl_indices], dim=1)
            excl_values = torch.cat([excl_values, new_excl_values])
        ms.exclusions = torch.sparse_coo_tensor(
            excl_indices, excl_values, (new_Nm, ms.Nmicro), device=device
        )

        # 4. condition_lengths
        ms.condition_lengths = torch.cat([
            ms.condition_lengths,
            torch.tensor([len(condition_indices)], device=device)
        ])

        # 5. threshold
        new_threshold = initial_threshold if initial_threshold is not None else 0.5
        ms.threshold = torch.cat([
            ms.threshold,
            torch.tensor([new_threshold], device=device, dtype=dtype)
        ])

        # 6. active y activation_level
        ms.active = torch.cat([
            ms.active,
            torch.tensor([False], device=device)
        ])
        ms.activation_level = torch.cat([
            ms.activation_level,
            torch.tensor([0.0], device=device, dtype=dtype)
        ])

        # 7. transitions
        new_transitions = torch.full((new_Nm, new_Nm), 0.1, device=device, dtype=dtype)
        new_transitions[:ms.Nm, :ms.Nm] = ms.transitions
        ms.transitions = new_transitions

        # 8. historial (lista Python)
        ms.activation_history.append([])

        # 9. Actualizar Nm
        ms.Nm = new_Nm

        logger.debug(f"Nueva macro‑neurona añadida: {new_id}")
        return True