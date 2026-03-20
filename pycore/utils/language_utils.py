import re

def normalize_text(text):
    """Normaliza el texto a minúsculas y elimina caracteres no relevantes."""
    return text.lower().strip()

def levenshtein_distance(a, b):
    """Calcula la distancia de Levenshtein entre dos cadenas."""
    if len(a) < len(b):
        return levenshtein_distance(b, a)
    if len(b) == 0:
        return len(a)
    previous_row = range(len(b) + 1)
    for i, c1 in enumerate(a):
        current_row = [i + 1]
        for j, c2 in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def correct_word(word, vocabulary, max_dist=2):
    """
    Corrige una palabra usando distancia de Levenshtein sobre el vocabulario dado.
    Si no hay corrección cercana, retorna la palabra original.
    """
    word = word.lower()
    if word in [v.lower() for v in vocabulary]:
        return word  # No corregir si ya está en el vocabulario
    best = word
    best_dist = max_dist + 1
    candidates = []
    for v in vocabulary:
        dist = levenshtein_distance(word, v.lower())
        if dist < best_dist:
            best = v
            best_dist = dist
            candidates = [v]
        elif dist == best_dist:
            candidates.append(v)
    # Solo corregir si la distancia es 1, o 2 si la palabra es larga (>6)
    if best_dist == 1 or (best_dist == 2 and len(word) > 6):
        if len(candidates) == 1:
            return best
    return word

def correct_phrase(phrase, vocabulary):
    """
    Corrige cada palabra de la frase usando el vocabulario dado.
    """
    words = re.findall(r'\w+', phrase, flags=re.UNICODE)
    corrected_words = [correct_word(w, vocabulary) for w in words]
    # Reconstruir la frase con espacios
    return ' '.join(corrected_words)