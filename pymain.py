#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main para probar la arquitectura cognitiva con personalidad Aura.
Soporte multilingüe (español/inglés) y aprendizaje por refuerzo.

Uso:
    python main.py [--device cpu|cuda] [--config ruta_yaml] [--lang es|en] [--no-response]

Ejemplo:
    python main.py --device cpu --lang es
"""

import os
import sys
import argparse
import torch
import yaml
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pycore
from pycore.core.memory import Memory
from pycore.core.personality import Personality
from pycore.core.cognitive_engine import CognitiveEngine
from pycore.loaders.neuron_loader import load_neurons
from pycore.deliberation.thinking_neurons import populate_tns
from pycore.language.grammar_adjudicator import GrammarAdjudicator
from pycore.deliberation.context_synthesizer import ContextSynthesizer
from pycore.deliberation.response_builder import ResponseBuilder
from pycore.inference.micro_inference import adjust_micro_thresholds
from pycore.inference.neuron_inference import adjust_neuron_thresholds


def main():
    parser = argparse.ArgumentParser(description='Ejecutar Aura AI')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Dispositivo para PyTorch (cpu o cuda)')
    parser.add_argument('--config', type=str, default='config/personality.yaml',
                        help='Ruta al archivo YAML de personalidad')
    parser.add_argument('--lang', type=str, default='es', choices=['es', 'en'],
                        help='Idioma de la conversación (español o inglés)')
    parser.add_argument('--no-response', action='store_true',
                        help='No generar respuesta (solo mostrar activaciones)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Usando dispositivo: {device}")
    print(f"Idioma seleccionado: {args.lang}")

    print("Archivo de memoria usado:", pycore.core.memory.__file__)

    # 1. Cargar personalidad
    if not os.path.exists(args.config):
        print(f"Error: No se encuentra el archivo de personalidad {args.config}")
        sys.exit(1)
    personality = Personality(args.config)

    # 2. Crear memoria y engine
    memory = Memory(device=device)
    engine = CognitiveEngine(memory, personality, device=device)

    # 3. Cargar neuronas (base, aprendidas, personalizadas)
    print("Cargando neuronas...")
    summary = load_neurons(engine, device=device)
    from pycore.utils.word_vectors import get_word_vectors
    _ = get_word_vectors()  # Esto carga los vectores y los mantiene en el singleton
    print("Vectores FastText cargados.")
    print(f"Micro-neuronas: {len(summary['micro_neurons'])}")
    print(f"Neuronas: {len(summary['neurons'])}")
    print(f"Macro-neuronas: {len(summary['macro_neurons'])}")
    print(f"Interconectoras: {len(summary['interconnectors'])}")

    # Mostrar las primeras macro‑neuronas (para depuración)
    if summary['macro_neurons']:
        print("Primeras macro‑neuronas:", summary['macro_neurons'][:5])

    # 4. Poblar Thinking Neurons y MacroTN (si se quiere respuesta)
    if not args.no_response:
        print("Inicializando Thinking Neurons...")
        tn_list, macro_tn = populate_tns(memory=memory,
                                        neural_system=engine,
                                        interconnectors=engine.interconnector_state,
                                        engine=engine)
        # Crear adjudicador gramatical, sintetizador de contexto y response builder
        adjudicator = GrammarAdjudicator()
        synthesizer = ContextSynthesizer(engine)
        response_builder = ResponseBuilder(
            personality=personality,
            context=None,
            winning_plan=None,
            adjudicator=adjudicator,
            engine=engine,
            interconnectors=engine.interconnector_state
        )
        # Establecer el idioma del response builder para la selección de palabras
        response_builder.language = args.lang
    else:
        macro_tn = None
        synthesizer = None
        response_builder = None

    # 5. Bucle conversacional
    print("\n" + "=" * 50)
    print("Sistema listo. Escribe 'salir' para terminar.")
    print("=" * 50)

    interaction_counter = 0
    while True:
        try:
            user_input = input("\nTú: ")
            if user_input.lower() in ('salir', 'exit', 'quit'):
                break

            # Generar embedding de la entrada (simulado)
            if engine.micro_state is not None and engine.micro_state.n > 0:
                from pycore.states.micro_state import compute_embedding
                from pycore.utils.word_vectors import get_word_vectors
                word_vec = get_word_vectors()
                emb = word_vec.embed_text(user_input).to(device)
                input_vectors = emb
            else:
                print("Error: No hay micro‑neuronas cargadas.")
                continue
            print("[DEBUG] Antes de llamar a engine.iterative_process_input")
            # Ejecutar ciclo de razonamiento
            final_state = engine.iterative_process_input(
                input_vectors,
                original_phrase=user_input,
                micro_threshold=0.7,
                num_iterations=10
            )
            print("[DEBUG] Después de iterative_process_input")
            # Depuración: mostrar macro‑neuronas activas (si existen)
            if engine.macro_state is not None:
                active_macros = engine.macro_state.active.sum().item()
                if active_macros > 0:
                    print(f"[DEBUG] Macro‑neuronas activas: {active_macros}")

            # Extraer IDs activas para feedback
            active_micro_ids = [mid for mid, act in final_state['neural_state']['micro_neurons'].items() if act]
            active_neuron_ids = [nid for nid, act in final_state['neural_state']['neurons'].items() if act]

            if args.no_response:
                # Mostrar solo activaciones finales
                print("\n[Activaciones finales]")
                print("Micro-neuronas activas:", len(active_micro_ids))
                print("Neuronas activas:", len(active_neuron_ids))
                if active_micro_ids:
                    print("IDs micro activas:", active_micro_ids[:10])
            else:
                # Construir neural_state para el sintetizador
                neural_state = {
                    'micro_neurons': final_state['neural_state']['micro_neurons'],
                    'neurons': final_state['neural_state']['neurons']
                }
                retrieved = {}

                # Sintetizar hipótesis de contexto
                context_hypotheses = synthesizer.synthesize(neural_state, retrieved)
                print(f"[DEBUG Main] Context hypotheses generadas: {len(context_hypotheses)}")
                for i, h in enumerate(context_hypotheses):
                    print(f"  {i}: type={h.get('type')}, subtype={h.get('subtype')}, confidence={h.get('confidence')}")

                # Obtener plan de MacroTN
                winning_plan, _, _ = macro_tn.reasoning_cycle(neural_state, retrieved, context_hypotheses)

                if winning_plan is not None:
                    # Depuración: mostrar el plan conceptual
                    print(f"[DEBUG] Conceptual plan: {winning_plan.get('conceptual_plan')}")
                    response_builder.winning_plan = winning_plan
                    response = response_builder.build_response()
                else:
                    response = "No estoy seguro de cómo responder."

                print(f"\nAura: {response}")

                # --- Feedback del usuario (aprendizaje por refuerzo) ---
                feedback = input("¿Te ha parecido útil la respuesta? (s/n): ").lower()
                if feedback in ('s', 'n'):
                    success = (feedback == 's')
                    engine.reinforce_proposal(
                        successful_proposal=success,
                        active_micro_ids=active_micro_ids,
                        active_neuron_ids=active_neuron_ids,
                        success_rate=0.05,
                        failure_rate=-0.02
                    )
                    print("  (Refuerzo aplicado)" if success else "  (Debilitamiento aplicado)")

                # --- Homeostasis periódica (cada 10 interacciones) ---
                interaction_counter += 1
                if interaction_counter % 10 == 0:
                    if engine.micro_state:
                        engine.micro_state = adjust_micro_thresholds(engine.micro_state)
                    if engine.neuron_state:
                        engine.neuron_state = adjust_neuron_thresholds(engine.neuron_state)
                    print("  (Homeostasis aplicada)")
                    if engine.micro_state:
                        print(f"    Umbral micro medio: {engine.micro_state.activation_threshold.mean().item():.3f}")
                    if engine.neuron_state:
                        print(f"    Umbral neurona medio: {engine.neuron_state.activation_threshold.mean().item():.3f}")

        except KeyboardInterrupt:
            print("\nInterrupción detectada.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

    print("Sistema finalizado.")


if __name__ == '__main__':
    main() 