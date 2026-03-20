# homeostatic.py
import torch
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)
# learning/homeostatic.py
# Por ahora, moveremos las funciones de ajuste homeostático desde inference/ a este módulo.
# Las funciones ya están en micro_inference y neuron_inference, pero podemos reexportarlas aquí
# para que CognitiveEngine las use desde learning. En fase 2 las moveremos completamente.

from ..inference.micro_inference import adjust_micro_thresholds
from ..inference.neuron_inference import adjust_thresholds as adjust_neuron_thresholds

__all__ = ['adjust_micro_thresholds', 'adjust_neuron_thresholds']

# Función a borrar cuando se integre completamente en el engine
def meta_adjust_parameters_(self, window: int = None):
    """
    Ajusta umbrales y tasas de decaimiento según el historial reciente.
    
    Args:
        window: Número de ciclos a considerar para el ajuste (si None, usa valor de configuración)
    """
    # Usar ventana de configuración si no se proporciona
    if window is None:
        window = self.config.model.engine.meta_adjust.window
        
    logger.debug(f"Iniciando meta-ajuste con ventana {window}")
    
    if len(self.cycle_history) < window:
        logger.debug(f"Historial insuficiente: {len(self.cycle_history)} < {window}")
        return
        
    recent = self.cycle_history[-window:]
    logger.debug(f"Analizando {len(recent)} ciclos recientes")

    # Obtener parámetros de configuración para meta-ajuste
    low_threshold = self.config.model.engine.meta_adjust.low_threshold
    high_threshold = self.config.model.engine.meta_adjust.high_threshold
    threshold_adjust = self.config.model.engine.meta_adjust.threshold_adjust
    decay_adjust = self.config.model.engine.meta_adjust.decay_adjust
    
    logger.debug(f"Parámetros meta-ajuste: bajo={low_threshold}, alto={high_threshold}, "
                f"ajuste_umbral={threshold_adjust}, ajuste_decay={decay_adjust}")

    # Micro‑neuronas
    if self.micro_state:
        logger.debug(f"Ajustando {self.micro_state.n} micro-neuronas")
        adjustments_made = 0
        for i in range(self.micro_state.n):
            mid = self.micro_state.ids[i]
            activations = [c['micro_activations'][mid]['activation_level']
                            for c in recent if mid in c['micro_activations']]
            if activations:
                avg = sum(activations) / len(activations)
                if avg < low_threshold:
                    self.micro_state.activation_threshold[i] = max(0.1, self.micro_state.activation_threshold[i] - threshold_adjust)
                    self.micro_state.decay_rate[i] = max(0.01, self.micro_state.decay_rate[i] - decay_adjust)
                    logger.debug(f"  - Micro {mid}: avg={avg:.3f} < {low_threshold} -> threshold={self.micro_state.activation_threshold[i].item():.3f}, decay={self.micro_state.decay_rate[i].item():.3f}")
                    adjustments_made += 1
                elif avg > high_threshold:
                    self.micro_state.activation_threshold[i] = min(1.0, self.micro_state.activation_threshold[i] + threshold_adjust)
                    self.micro_state.decay_rate[i] = min(1.0, self.micro_state.decay_rate[i] + decay_adjust)
                    logger.debug(f"  - Micro {mid}: avg={avg:.3f} > {high_threshold} -> threshold={self.micro_state.activation_threshold[i].item():.3f}, decay={self.micro_state.decay_rate[i].item():.3f}")
                    adjustments_made += 1
        logger.debug(f"Ajustes realizados en micro-neuronas: {adjustments_made}")

    # Neuronas
    if self.neuron_state:
        logger.debug(f"Ajustando {self.neuron_state.N} neuronas intermedias")
        adjustments_made = 0
        for i in range(self.neuron_state.N):
            nid = self.neuron_state.ids[i]
            activations = [c['neuron_details'][nid]['activation_level']
                            for c in recent if nid in c['neuron_details']]
            if activations:
                avg = sum(activations) / len(activations)
                if avg < low_threshold:
                    self.neuron_state.activation_threshold[i] = max(0.1, self.neuron_state.activation_threshold[i] - threshold_adjust)
                    self.neuron_state.decay_rate[i] = max(0.01, self.neuron_state.decay_rate[i] - decay_adjust)
                    logger.debug(f"  - Neurona {nid}: avg={avg:.3f} < {low_threshold} -> threshold={self.neuron_state.activation_threshold[i].item():.3f}, decay={self.neuron_state.decay_rate[i].item():.3f}")
                    adjustments_made += 1
                elif avg > high_threshold:
                    self.neuron_state.activation_threshold[i] = min(1.0, self.neuron_state.activation_threshold[i] + threshold_adjust)
                    self.neuron_state.decay_rate[i] = min(1.0, self.neuron_state.decay_rate[i] + decay_adjust)
                    logger.debug(f"  - Neurona {nid}: avg={avg:.3f} > {high_threshold} -> threshold={self.neuron_state.activation_threshold[i].item():.3f}, decay={self.neuron_state.decay_rate[i].item():.3f}")
                    adjustments_made += 1
        logger.debug(f"Ajustes realizados en neuronas: {adjustments_made}")


def meta_adjust_parameters(engine, window: int = None):
    """
    Ajusta umbrales y tasas de decaimiento según el historial reciente.
    Esta función debe ser llamada desde el engine, pero la lógica se extrae aquí.
    """
    config = engine.config
    if window is None:
        window = config.model.engine.meta_adjust.window

    if len(engine.cycle_history) < window:
        logger.debug(f"Historial insuficiente: {len(engine.cycle_history)} < {window}")
        return

    recent = engine.cycle_history[-window:]

    low_threshold = config.model.engine.meta_adjust.low_threshold
    high_threshold = config.model.engine.meta_adjust.high_threshold
    threshold_adjust = config.model.engine.meta_adjust.threshold_adjust
    decay_adjust = config.model.engine.meta_adjust.decay_adjust

    # Micro‑neuronas
    if engine.micro_state:
        for i, mid in enumerate(engine.micro_state.ids):
            activations = [c['micro_activations'][mid]['activation_level']
                           for c in recent if mid in c['micro_activations']]
            if activations:
                avg = sum(activations) / len(activations)
                if avg < low_threshold:
                    engine.micro_state.activation_threshold[i] = max(0.1, engine.micro_state.activation_threshold[i] - threshold_adjust)
                    engine.micro_state.decay_rate[i] = max(0.01, engine.micro_state.decay_rate[i] - decay_adjust)
                elif avg > high_threshold:
                    engine.micro_state.activation_threshold[i] = min(1.0, engine.micro_state.activation_threshold[i] + threshold_adjust)
                    engine.micro_state.decay_rate[i] = min(1.0, engine.micro_state.decay_rate[i] + decay_adjust)

    # Neuronas
    if engine.neuron_state:
        for i, nid in enumerate(engine.neuron_state.ids):
            activations = [c['neuron_details'][nid]['activation_level']
                           for c in recent if nid in c['neuron_details']]
            if activations:
                avg = sum(activations) / len(activations)
                if avg < low_threshold:
                    engine.neuron_state.activation_threshold[i] = max(0.1, engine.neuron_state.activation_threshold[i] - threshold_adjust)
                    engine.neuron_state.decay_rate[i] = max(0.01, engine.neuron_state.decay_rate[i] - decay_adjust)
                elif avg > high_threshold:
                    engine.neuron_state.activation_threshold[i] = min(1.0, engine.neuron_state.activation_threshold[i] + threshold_adjust)
                    engine.neuron_state.decay_rate[i] = min(1.0, engine.neuron_state.decay_rate[i] + decay_adjust)