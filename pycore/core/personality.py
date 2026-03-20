import yaml
import torch
from typing import Dict, Any, Optional

class Personality:
    """
    Gestor de la personalidad de la IA.
    Carga la configuración desde un archivo YAML y la mantiene disponible.
    Se pueden obtener representaciones tensoriales de los rasgos si es necesario.
    """
    def __init__(self, yaml_path: str):
        """
        Carga la personalidad desde un archivo YAML.
        Ejemplo de contenido YAML:
            name: Krystal
            traits:
                formality: 0.6
                empathy: 0.7
                logical_analysis: 0.9
                creativity: 0.6
            emotional_state:
                mood: neutral
                intensity: 0.5
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        self.name: str = data.get('name', 'IA')
        self.traits: Dict[str, Any] = data.get('traits', {})
        self.emotional_state: Dict[str, Any] = data.get('emotional_state', {'mood': 'neutral', 'intensity': 0.5})

    def traits_to_tensor(self, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        Convierte los rasgos de personalidad a un tensor 1D.
        Solo se incluyen aquellos rasgos que tengan valores numéricos (float/int).
        El orden está determinado por la clave en el diccionario (sorted).
        """
        values = []
        for k in sorted(self.traits.keys()):
            v = self.traits[k]
            if isinstance(v, (int, float)):
                values.append(float(v))
        if not values:
            return torch.tensor([], device=device)
        return torch.tensor(values, device=device)

    def emotional_state_to_tensor(self, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        Convierte el estado emocional a un tensor.
        Por ahora, solo la intensidad (float) se puede usar.
        El humor se podría mapear a one‑hot si se desea.
        """
        intensity = float(self.emotional_state.get('intensity', 0.5))
        return torch.tensor([intensity], device=device)

    def __repr__(self) -> str:
        return f"Personality(name={self.name}, traits={self.traits}, emotional_state={self.emotional_state})" 