import nltk
import numpy as np
import pandas as pd
import uuid
import re
from typing import List, Dict, Any, Tuple
from transformers import PreTrainedTokenizer

# Télécharger les ressources de segmentation de phrase
nltk.download("punkt")

def count_tokens(text: str, tokenizer: PreTrainedTokenizer, max_len=512) -> int:
    """Compte le nombre de tokens selon le tokenizer donné."""
    return len(tokenizer.tokenize(text))

def split_text_by_tokens(text: str, tokenizer: PreTrainedTokenizer, max_tokens: int = 300, overlap_tokens: int = 50) -> List[Dict]:
    """
    Découpe un texte en segments par tokens avec garantie de respecter max_tokens.
    """
    if not text.strip():
        return []
    
    # Tokenisation avec offsets
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    tokens = encoding['input_ids']
    offset_mapping = encoding['offset_mapping']
    
    if not tokens:
        return []
    
    # Si le texte est plus court que max_tokens, retourner tel quel
    if len(tokens) <= max_tokens:
        return [{
            "text": text,
            "start": 0,
            "end": len(text)
        }]
    
    segments = []
    i = 0
    
    while i < len(tokens):
        # GARANTIR que le segment ne dépasse jamais max_tokens
        end_token_idx = min(i + max_tokens, len(tokens))
        
        # Extraire les positions de caractères
        start_char = offset_mapping[i][0]
        end_char = offset_mapping[end_token_idx - 1][1]
        
        segment_text = text[start_char:end_char]
        
        # VÉRIFICATION DOUBLE : compter les tokens du segment
        segment_token_count = len(tokenizer.tokenize(segment_text))
        
        # Si le segment est encore trop long, le raccourcir
        while segment_token_count > max_tokens and end_token_idx > i + 1:
            end_token_idx -= 1
            end_char = offset_mapping[end_token_idx - 1][1]
            segment_text = text[start_char:end_char]
            segment_token_count = len(tokenizer.tokenize(segment_text))
        
        segments.append({
            "text": segment_text,
            "start": start_char,
            "end": end_char
        })
        
        # Avancer avec overlap
        if end_token_idx >= len(tokens):
            break
            
        # Calculer la position suivante avec overlap
        next_i = max(end_token_idx - overlap_tokens, i + 1)
        i = next_i
    
    return segments

def extract_agreed_entities_labelstudio(
    articles: List[str],
    gliner,
    nuner,
    tokenizer,
    labels: List[str],
    max_articles: int = None,
    max_tokens: int = 300,
    overlap_tokens: int = 50
) -> Tuple[List[Dict], int, int]:
    """
    Version corrigée qui garantit de ne jamais dépasser max_tokens.
    """
    labelstudio_data = []
    total_agreements = 0
    total_disagreements = 0

    for idx, full_text in enumerate(articles):
        if not isinstance(full_text, str) or not full_text.strip():
            continue
        if max_articles and idx >= max_articles:
            break

        try:
            # Utiliser la segmentation corrigée
            segments = split_text_by_tokens(full_text, tokenizer, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        except Exception as e:
            print(f"[!] Segmentation échouée article {idx}: {e}")
            continue

        agreed_entities = []
        processed_entities = set()
        article_agreements = 0
        article_disagreements = 0

        for segment in segments:
            segment_text = segment["text"]
            offset = segment["start"]
            
            # VÉRIFICATION FINALE avant prédiction
            segment_token_count = len(tokenizer.tokenize(segment_text))
            if segment_token_count > max_tokens:
                print(f"[!] ERREUR: Segment encore trop long ({segment_token_count} tokens), truncature forcée")
                segment_text = truncate_text_to_tokens(segment_text, tokenizer, max_tokens)
                
            try:
                gliner_ents = gliner.predict_entities(segment_text, labels, threshold=0.4)
                nuner_ents = nuner.predict_entities(segment_text, labels, threshold=0.4)
            except Exception as e:
                print(f"[!] Erreur prédiction article {idx}, segment {offset}: {e}")
                continue

            # Reste du code identique...
            segment_agreements = 0
            gliner_matched = set()
            nuner_matched = set()
            
            for i, g in enumerate(gliner_ents):
                for j, x in enumerate(nuner_ents):
                    if (
                        abs(g["start"] - x["start"]) <= 3 and
                        abs(g["end"] - x["end"]) <= 3 and
                        g["label"] == x["label"]
                    ):
                        abs_start = offset + g["start"]
                        abs_end = offset + g["end"]
                        entity_key = (abs_start, abs_end, g["label"])
                        
                        if entity_key not in processed_entities:
                            agreed_entities.append({
                                "start": abs_start,
                                "end": abs_end,
                                "text": full_text[abs_start:abs_end],
                                "label": g["label"]
                            })
                            processed_entities.add(entity_key)
                            segment_agreements += 1
                            
                        gliner_matched.add(i)
                        nuner_matched.add(j)
                        break
            
            gliner_unmatched = len(gliner_ents) - len(gliner_matched)
            nuner_unmatched = len(nuner_ents) - len(nuner_matched)
            segment_disagreements = gliner_unmatched + nuner_unmatched
            
            article_agreements += segment_agreements
            article_disagreements += segment_disagreements

        total_agreements += article_agreements
        total_disagreements += article_disagreements

        if not agreed_entities:
            continue

        result_entries = [{
            "value": {
                "start": ent["start"],
                "end": ent["end"],
                "text": ent["text"],
                "labels": [ent["label"]]
            },
            "type": "labels",
            "from_name": "label",
            "to_name": "text",
            "id": str(uuid.uuid4())
        } for ent in agreed_entities]

        labelstudio_data.append({
            "id": str(uuid.uuid4()),
            "data": {
                "text": full_text
            },
            "predictions": [{
                "result": result_entries
            }]
        })

    return labelstudio_data, total_agreements, total_disagreements


def extract_entities_labelstudio(
    articles: List[str],
    model,
    tokenizer,
    labels: List[str],
    max_articles: int = None,
    max_tokens: int = 300,
    overlap_tokens: int = 50,
    threshold: float = 0.4
) -> Tuple[List[Dict], int]:
    """
    Version pour un seul modèle de prédiction d'entités.
    
    Args:
        articles: Liste des textes à traiter
        model: Modèle de prédiction (GLiNER ou NuNER)
        tokenizer: Tokenizer pour la segmentation
        labels: Liste des labels à prédire
        max_articles: Nombre maximum d'articles à traiter
        max_tokens: Nombre maximum de tokens par segment
        overlap_tokens: Nombre de tokens de chevauchement
        threshold: Seuil de confiance pour les prédictions
        
    Returns:
        Tuple contenant:
        - Liste des données au format LabelStudio
        - Nombre total d'entités extraites
    """
    labelstudio_data = []
    total_entities = 0

    for idx, full_text in enumerate(articles):
        if not isinstance(full_text, str) or not full_text.strip():
            continue
        if max_articles and idx >= max_articles:
            break

        try:
            # Utiliser la segmentation corrigée
            segments = split_text_by_tokens(
                full_text, 
                tokenizer, 
                max_tokens=max_tokens, 
                overlap_tokens=overlap_tokens
            )
        except Exception as e:
            print(f"[!] Segmentation échouée article {idx}: {e}")
            continue

        all_entities = []
        processed_entities = set()
        article_entity_count = 0

        for segment in segments:
            segment_text = segment["text"]
            offset = segment["start"]
            
            # Vérification finale avant prédiction
            segment_token_count = len(tokenizer.tokenize(segment_text))
            if segment_token_count > max_tokens:
                print(f"[!] ERREUR: Segment trop long ({segment_token_count} tokens), truncature forcée")
                segment_text = truncate_text_to_tokens(segment_text, tokenizer, max_tokens)
                
            try:
                predicted_entities = model.predict_entities(
                    segment_text, 
                    labels, 
                    threshold=threshold
                )
            except Exception as e:
                print(f"[!] Erreur prédiction article {idx}, segment {offset}: {e}")
                continue

            # Traiter les entités prédites
            for entity in predicted_entities:
                abs_start = offset + entity["start"]
                abs_end = offset + entity["end"]
                entity_key = (abs_start, abs_end, entity["label"])
                
                # Éviter les doublons dus au chevauchement
                if entity_key not in processed_entities:
                    all_entities.append({
                        "start": abs_start,
                        "end": abs_end,
                        "text": full_text[abs_start:abs_end],
                        "label": entity["label"]
                    })
                    processed_entities.add(entity_key)
                    article_entity_count += 1

        total_entities += article_entity_count

        if not all_entities:
            continue

        # Formater pour LabelStudio
        result_entries = [{
            "value": {
                "start": ent["start"],
                "end": ent["end"],
                "text": ent["text"],
                "labels": [ent["label"]]
            },
            "type": "labels",
            "from_name": "label",
            "to_name": "text",
            "id": str(uuid.uuid4())
        } for ent in all_entities]

        labelstudio_data.append({
            "id": str(uuid.uuid4()),
            "data": {
                "text": full_text
            },
            "predictions": [{
                "result": result_entries
            }]
        })

    return labelstudio_data, total_entities