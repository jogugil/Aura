"""
Loader y gestor de neuronas para arquitectura modular (versión PyTorch con estados agrupados).
- Carga neuronas básicas, aprendidas y personalizadas.
- Prioriza: personalizadas > aprendidas > base.
- Construye los estados agrupados y los asigna al motor cognitivo (engine).
- Ahora utiliza embeddings semánticos (FastText) para las micro‑neuronas.
- Permite fácil edición y limpieza de neuronas aprendidas (guardado en JSON).
"""

import json
import os
import torch
import hashlib
import logging
from typing import Dict, List, Any, Optional

# Configurar logger (nivel INFO para ver mensajes importantes)
logger = logging.getLogger(__name__)

# Importaciones de los módulos de datos (usando rutas absolutas desde pycore)
from ..data.base_neuron_data import get_base_data
from ..data.learned_neuron_data import get_learned_neurons

# Importar el cargador de vectores semánticos (FastText)
from ..utils.word_vectors import get_word_vectors

# Intentar importar personalizadas (puede no existir)
try:
    from ..data.personality_neurons import get_personality_neurons
except ImportError:
    get_personality_neurons = lambda: []

# Importaciones de los estados
from ..states.micro_state import MicroNeuronState
from ..states.neuron_state import NeuronState, create_neuron_state_from_ids
from ..states.macro_state import MacroNeuronState, create_macro_state_from_ids
from ..states.interconnector_state import InterconnectorState, create_interconnector_state_from_ids
from ..core.cognitive_engine import CognitiveEngine


# ----------------------------------------------------------------------
# Función auxiliar para generar prototipos de conceptos (offline con caché)
# ----------------------------------------------------------------------
def _get_cache_path(micro_ids: List[str], micro_concepts: List[str]) -> str:
    """
    Genera un nombre de archivo único basado en los IDs y conceptos.
    """
    combined = "".join(micro_ids) + "".join(micro_concepts)
    hash_id = hashlib.md5(combined.encode()).hexdigest()
    return os.path.join("cache", f"micro_embeddings_{hash_id}.pt")


def generate_prototypes(micro_ids: List[str], micro_concepts: List[str]) -> torch.Tensor:
    """
    Genera embeddings prototipo para cada micro‑neurona usando FastText.
    Utiliza caché en disco para evitar recalcular cada vez.
    """
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = _get_cache_path(micro_ids, micro_concepts)

    # Si ya existe, cargar y devolver
    if os.path.exists(cache_path):
        logger.info(f"📦 Cargando prototipos desde caché: {cache_path}")
        try:
            embeddings = torch.load(cache_path, map_location='cpu')
            logger.info(f"✅ Embeddings cargados: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.warning(f"⚠️ Error al cargar caché ({e}). Regenerando...")
            # Si falla, continuamos con la generación (no hacemos return)

    # Si no existe o falló la carga, generar desde cero
    logger.info("🔨 Generando prototipos desde FastText (puede tardar un poco)...")

    # Obtener el cargador de vectores (singleton)
    word_vec = get_word_vectors()
    dim = word_vec.dim
    logger.info(f"📐 Dimensión de FastText: {dim}")

    # Mapeo de frases representativas para cada concepto
    CONCEPT_PHRASES = {
        # Conceptos sociales
        "saludo": ["hola", "buenos días", "saludos", "hello", "good morning"],
        "despedida": ["adiós", "hasta luego", "bye", "goodbye"],
        "pregunta_bienestar": ["cómo estás", "qué tal", "how are you"],
        "respuesta_bienestar": ["bien", "muy bien", "estoy bien", "I'm fine"],
        "agradecimiento": ["gracias", "thank you", "thanks"],
        "acuerdo": ["sí", "claro", "ok", "yes", "okay"],
        "desacuerdo": ["no", "nunca", "no way", "nope"],
        "pregunta_identidad": ["quién eres", "who are you"],
        "pregunta_capacidad": ["qué puedes hacer", "what can you do"],
        "pregunta_tiempo": ["qué hora es", "what time is it"],
        "pregunta_lugar": ["dónde estás", "where are you"],
        "pregunta_razon": ["por qué", "why"],
        "numero": ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        "operacion_suma": ["más", "plus", "suma", "+"],
        "operacion_resta": ["menos", "minus", "resta", "-"],
        "operacion_multiplicacion": ["por", "times", "multiplicación"],
        "operacion_division": ["dividido", "divided by", "entre"],
        "pregunta_calculo": ["cuánto es", "how much is"],
        "resultado_calculo": ["resultado", "result"],
        "clarificacion": ["aclara", "explícame", "clarify", "explain"],
        "iniciar_conversacion": ["empezar", "comencemos", "let's start"],
        "ofrecer_ayuda": ["ayuda", "help"],
        "clima": ["clima", "tiempo", "weather"],
        "tiempo": ["tiempo", "hora", "time"],
        "identidad_externa": ["tú", "you"],
    }

    prototypes = []
    total = len(micro_ids)
    for i, mid in enumerate(micro_ids):
        if (i+1) % 50 == 0:
            logger.info(f"  Progreso: {i+1}/{total} conceptos procesados")
        phrases = CONCEPT_PHRASES.get(mid, [mid])
        phrase_vectors = []
        for phrase in phrases:
            vec = word_vec.embed_text(phrase)
            phrase_vectors.append(vec)
        if phrase_vectors:
            prototype = torch.stack(phrase_vectors).mean(dim=0)
        else:
            logger.warning(f"⚠️ No hay frases para el concepto {mid}, usando vector aleatorio")
            prototype = torch.randn(dim)
            prototype = prototype / prototype.norm()
        prototypes.append(prototype)

    embeddings = torch.stack(prototypes)
    logger.info(f"✅ Prototipos generados: {embeddings.shape}")

    # Guardar en caché
    logger.info(f"💾 Guardando prototipos en caché: {cache_path}")
    torch.save(embeddings, cache_path)

    return embeddings


# ----------------------------------------------------------------------
# Función principal de carga de neuronas
# ----------------------------------------------------------------------
def load_neurons(engine: CognitiveEngine, device=None) -> Dict[str, List[str]]:
    """
    Carga todas las neuronas desde las fuentes base, aprendidas y personalizadas,
    construye los estados y los asigna al motor cognitivo (engine).

    Args:
        engine: instancia de CognitiveEngine (debe tener memoria y personalidad).
        device: dispositivo para los tensores (si None, se usa el del engine).

    Returns:
        Un diccionario con las listas de IDs clasificadas (para compatibilidad):
        {
            "micro_neurons": lista de IDs,
            "neurons": lista de IDs,
            "macro_neurons": lista de IDs,
            "interconnectors": lista de IDs,
            "all": lista de IDs (todas las neuronas)
        }
    """
    if device is None:
        device = engine.device

    # ------------------------------------------------------------
    # 1. Obtener datos base
    # ------------------------------------------------------------
    base_data = get_base_data()

    # ------------------------------------------------------------
    # 2. Obtener datos aprendidos y personalizados
    # ------------------------------------------------------------
    learned = get_learned_neurons()          # lista de dicts
    personality = get_personality_neurons()  # lista de dicts

    # ------------------------------------------------------------
    # 3. Fusionar con prioridad: personality > learned > base
    # ------------------------------------------------------------
    # Diccionarios para cada tipo de entidad, clave = ID
    micro_dict = {}      # id -> (concept, type, metadata)
    neuron_dict = {}     # id -> (name, condition_ids, exclusion_ids)
    macro_dict = {}      # id -> (name, condition_n_ids, exclusion_mn_ids, metadata)
    inter_dict = {}      # id -> (connected_neuron_ids, rules, metadata)

    # 3a. Cargar base: micro y neuronas
    for i, mid in enumerate(base_data['micro_ids']):
        micro_dict[mid] = (base_data['micro_concepts'][i],
                           base_data['micro_types'][i],
                           base_data['micro_metadata'][i])
    for i, nid in enumerate(base_data['neuron_ids']):
        neuron_dict[nid] = (base_data['neuron_names'][i],
                            base_data['neuron_conditions'][i],
                            base_data['neuron_exclusions'][i])

    # 3b. Cargar base: macro‑neuronas (si existen en base_data)
    if 'macro_ids' in base_data:
        for i, mid in enumerate(base_data['macro_ids']):
            name = base_data['macro_names'][i]
            condition_ids = base_data['macro_conditions'][i] if 'macro_conditions' in base_data else []
            exclusion_ids = base_data['macro_exclusions'][i] if 'macro_exclusions' in base_data else []
            metadata = base_data['macro_metadata'][i] if 'macro_metadata' in base_data else {}
            macro_dict[mid] = (name, condition_ids, exclusion_ids, metadata)

    # 3c. Añadir aprendidas (sobrescriben)
    for item in learned:
        tid = item['id']
        neuron_type = item.get('type', 'keyword')
        if neuron_type in ('keyword', 'abstract_concept', 'pattern'):
            micro_dict[tid] = (item['concept'], neuron_type, item.get('metadata', {}))
        elif neuron_type == 'neuron':
            conds = item.get('condition_micro_ids', [])
            excs = item.get('exclusion_micro_ids', [])
            neuron_dict[tid] = (item.get('name', tid), conds, excs)
        elif neuron_type == 'macro':
            macro_dict[tid] = (item.get('name', tid),
                               item.get('condition_neuron_ids', []),
                               item.get('exclusion_micro_ids', []),
                               item.get('metadata', {}))
        elif neuron_type == 'interconnector':
            inter_dict[tid] = (item.get('connected_neuron_ids', []),
                               item.get('rules', {}),
                               item.get('metadata', {}))
        else:
            micro_dict[tid] = (item['concept'], neuron_type, item.get('metadata', {}))

    # 3d. Añadir personalizadas (sobrescriben) - misma lógica
    for item in personality:
        tid = item['id']
        neuron_type = item.get('type', 'keyword')
        if neuron_type in ('keyword', 'abstract_concept', 'pattern'):
            micro_dict[tid] = (item['concept'], neuron_type, item.get('metadata', {}))
        elif neuron_type == 'neuron':
            conds = item.get('condition_micro_ids', [])
            excs = item.get('exclusion_micro_ids', [])
            neuron_dict[tid] = (item.get('name', tid), conds, excs)
        elif neuron_type == 'macro':
            macro_dict[tid] = (item.get('name', tid),
                               item.get('condition_neuron_ids', []),
                               item.get('exclusion_micro_ids', []),
                               item.get('metadata', {}))
        elif neuron_type == 'interconnector':
            inter_dict[tid] = (item.get('connected_neuron_ids', []),
                               item.get('rules', {}),
                               item.get('metadata', {}))
        else:
            micro_dict[tid] = (item['concept'], neuron_type, item.get('metadata', {}))

    # ------------------------------------------------------------
    # 4. Clasificar y preparar listas para cada estado
    # ------------------------------------------------------------
    micro_ids, micro_concepts, micro_types, micro_metadata = [], [], [], []
    neuron_ids, neuron_names, neuron_conditions, neuron_exclusions = [], [], [], []
    macro_ids, macro_names, macro_condition_n, macro_exclusion_mn, macro_metadata = [], [], [], [], []
    inter_ids, inter_connected, inter_rules, inter_metadata = [], [], [], []

    # Micro
    for mid, (conc, n_type, meta) in micro_dict.items():
        micro_ids.append(mid)
        micro_concepts.append(conc)
        micro_types.append(n_type)
        micro_metadata.append(meta)

    # Neuronas
    for nid, (name, conds, excs) in neuron_dict.items():
        neuron_ids.append(nid)
        neuron_names.append(name)
        neuron_conditions.append(conds)
        neuron_exclusions.append(excs)

    # Macro
    for mid, (name, conds_n, excs_mn, meta) in macro_dict.items():
        macro_ids.append(mid)
        macro_names.append(name)
        macro_condition_n.append(conds_n)
        macro_exclusion_mn.append(excs_mn)
        macro_metadata.append(meta)

    # Interconectoras
    for iid, (connected, rules, meta) in inter_dict.items():
        inter_ids.append(iid)
        inter_connected.append(connected)
        inter_rules.append(rules)
        inter_metadata.append(meta)

    # ------------------------------------------------------------
    # 5. Construir estados y asignar al engine
    # ------------------------------------------------------------

    # ------------------------------------------------------------------
    # 5.1 Micro‑neuronas con embeddings semánticos (FastText)
    # ------------------------------------------------------------------
    # Generamos los prototipos a partir de los IDs de micro (que deben ser nombres de concepto)
    micro_embeddings = generate_prototypes(micro_ids, micro_concepts).to(device)

    micro_state = MicroNeuronState(
        ids=micro_ids,
        concepts=micro_concepts,
        types=micro_types,
        metadata=micro_metadata,
        embeddings=micro_embeddings,
        device=device
    )
    engine.set_micro_state(micro_state)

    # Mapeo micro ID -> índice (necesario para neuronas y macro)
    micro_id_to_idx = {mid: i for i, mid in enumerate(micro_ids)}

    # ------------------------------------------------------------------
    # 5.2 Neuronas (nivel intermedio)
    # ------------------------------------------------------------------
    neuron_state = create_neuron_state_from_ids(
        neuron_ids=neuron_ids,
        names=neuron_names,
        condition_ids=neuron_conditions,
        exclusion_ids=neuron_exclusions,
        micro_id_to_index=micro_id_to_idx,
        num_micro=len(micro_ids),
        device=device
    )
    engine.set_neuron_state(neuron_state)

    # Mapeo neurona ID -> índice (para macro)
    neuron_id_to_idx = {nid: i for i, nid in enumerate(neuron_ids)}

    # ------------------------------------------------------------------
    # 5.3 Macro‑neuronas (si hay)
    # ------------------------------------------------------------------
    if macro_ids:
        macro_condition_indices = []
        for conds_ids in macro_condition_n:
            # Convertir IDs de neuronas a índices
            idx_list = [neuron_id_to_idx[cid] for cid in conds_ids if cid in neuron_id_to_idx]
            macro_condition_indices.append(idx_list)

        macro_exclusion_indices = []
        for excs_ids in macro_exclusion_mn:
            # Convertir IDs de micro a índices
            idx_list = [micro_id_to_idx[eid] for eid in excs_ids if eid in micro_id_to_idx]
            macro_exclusion_indices.append(idx_list)

        macro_state = create_macro_state_from_ids(
            macro_ids=macro_ids,
            names=macro_names,
            condition_indices=macro_condition_indices,
            exclusion_indices=macro_exclusion_indices,
            neuron_id_to_index=neuron_id_to_idx,
            micro_id_to_index=micro_id_to_idx,
            device=device
        )
    else:
        macro_state = MacroNeuronState(
            ids=[],
            names=[],
            condition_indices=[],
            exclusion_indices=[],
            device=device
        )
    engine.set_macro_state(macro_state)

    # ------------------------------------------------------------------
    # 5.4 Interconectoras (si hay)
    # ------------------------------------------------------------------
    if inter_ids:
        # Construir mapeo global de conceptos (IDs de cualquier tipo)
        global_concept_to_idx = {}
        offset = 0
        for lst, id_list in [(micro_ids, micro_ids), (neuron_ids, neuron_ids), (macro_ids, macro_ids)]:
            for i, cid in enumerate(id_list):
                global_concept_to_idx[cid] = offset + i
            offset += len(id_list)

        inter_state = create_interconnector_state_from_ids(
            inter_ids=inter_ids,
            connected_concept_ids=inter_connected,
            global_concept_to_idx=global_concept_to_idx,
            rules=inter_rules,
            device=device
        )
    else:
        inter_state = InterconnectorState(
            ids=[],
            connected_concept_ids=[],
            global_concept_to_idx={},
            device=device
        )
    engine.set_interconnector_state(inter_state)

    # ------------------------------------------------------------
    # 6. Devolver resumen (para compatibilidad)
    # ------------------------------------------------------------
    all_ids = micro_ids + neuron_ids + macro_ids + inter_ids
    return {
        "micro_neurons": micro_ids,
        "neurons": neuron_ids,
        "macro_neurons": macro_ids,
        "interconnectors": inter_ids,
        "all": all_ids
    }


# --- Utilidad para guardar una neurona aprendida en archivo JSON ---

def save_learned_neuron(neuron_dict: Dict[str, Any], file_path: str = 'learned_neurons.json'):
    """
    Guarda una neurona aprendida (como diccionario) en un archivo JSON.
    Si el archivo no existe, lo crea con una lista vacía.
    Luego se puede cargar con load_learned_neurons_json.
    """
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = []

    data.append(neuron_dict)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"💾 Neurona aprendida guardada en {file_path}: {neuron_dict.get('id', 'sin_id')}")


def load_learned_neurons_json(file_path: str = 'learned_neurons.json') -> List[Dict]:
    """Carga neuronas aprendidas desde un archivo JSON (si existe)."""
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# --- Función para migrar desde el formato antiguo (opcional) ---
def convert_from_objects(object_list):
    """
    Convierte una lista de objetos antiguos (MicroNeurona, Neurona, etc.)
    a una lista de diccionarios para usar con el nuevo loader.
    """
    result = []
    for obj in object_list:
        if hasattr(obj, 'to_dict'):
            result.append(obj.to_dict())
        else:
            # Intentar extraer atributos comunes
            d = {'id': obj.id, 'concept': getattr(obj, 'concepto', getattr(obj, 'name', '')), 'type': getattr(obj, 'type', 'desconocido')}
            if hasattr(obj, 'metadata'):
                d['metadata'] = obj.metadata
            else:
                d['metadata'] = {}
            result.append(d)
    return result