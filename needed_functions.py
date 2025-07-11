import json
import matplotlib.pyplot as plt
from collections import Counter
from functions import split_text_into_sliding_windows, create_tokenizer

from collections import defaultdict
import matplotlib.pyplot as plt

import random
from collections import defaultdict, Counter
import json


import re


def evaluate_ner(data, model_inference, classes, threshold=0.5, tolerance=10, 
                         segmentation_mode=0, overlap_ratio=0.1, verbose=True):
    """
    Version améliorée de l'évaluation NER avec matching déterministe et détails supplémentaires.
    """
    import time
    import numpy as np

    tp, fp, fn = 0, 0, 0
    class_metrics = {cls: {'tp': 0, 'fp': 0, 'fn': 0} for cls in classes}
    total_time = 0
    f1_by_doc = []
    
    # Statistiques additionnelles pour le débogage
    total_predictions = 0
    total_true_entities = 0
    
    # Préparer les données comme avant...
    prepared_data = []
    for item in data:
        text = item['data'].get('text')
        if not text:
            continue
        annotations = item.get('annotations', [])
        if not annotations:
            continue

        entities = []
        for ann in annotations[0]['result']:
            value = ann['value']
            ent_start = value['start']
            ent_end = value['end']
            label = value['labels'][0].lower()  # Convertir en minuscules
            entities.append((ent_start, ent_end, label))

        prepared_data.append((text, {'entities': entities}))

    # Évaluation principale
    for idx, (text, annotation) in enumerate(prepared_data):
        tp_i, fp_i, fn_i = 0, 0, 0
        start_time = time.time()

        words = text.split()
        if len(words) <= 350:
            predictions = model_inference.predict_entities(text, classes, threshold=threshold)
        else:
            predictions = []
            
            # Définir les paramètres de segmentation
            window_size = 350
            if segmentation_mode == 0:
                step_size = window_size
            else:
                step_size = int(window_size * (1 - overlap_ratio))
            
            # Traiter par segments avec déduplication
            all_segment_preds = []
            for i in range(0, len(words), step_size):
                end = min(i + window_size, len(words))
                segment = ' '.join(words[i:end])
                offset = 0 if i == 0 else len(' '.join(words[:i])) + 1

                segment_preds = model_inference.predict_entities(segment, classes, threshold=threshold)
                
                # Ajouter l'offset aux prédictions
                for pred in segment_preds:
                    pred_with_offset = {
                        'start': pred['start'] + offset,
                        'end': pred['end'] + offset,
                        'label': pred['label'].lower(),
                        'score': pred.get('score', 0.5)  # Inclure le score si disponible
                    }
                    all_segment_preds.append(pred_with_offset)
            
            # Déduplication des entités qui se chevauchent
            all_segment_preds.sort(key=lambda x: (x['score'], x['start']), reverse=True)
            
            # Garder les prédictions non chevauchantes avec priorité au score
            kept_preds = []
            for pred in all_segment_preds:
                # Vérifier chevauchement avec les prédictions déjà conservées
                overlaps = False
                for kept in kept_preds:
                    # Deux entités se chevauchent si l'une commence avant que l'autre ne finisse
                    if kept['label'] == pred['label'] and \
                       max(kept['start'], pred['start']) <= min(kept['end'], pred['end']):
                        overlaps = True
                        break
                
                if not overlaps:
                    kept_preds.append(pred)
            
            predictions = kept_preds

        total_time += time.time() - start_time
        
        # Statistiques pour debugging
        total_predictions += len(predictions)
        total_true_entities += len(annotation['entities'])

        # Tri déterministe des entités pour matching cohérent
        true_entities = sorted([(start, end, label) for start, end, label in annotation['entities']], 
                               key=lambda x: (x[0], x[1], x[2]))
        pred_entities = sorted([(p['start'], p['end'], p['label']) for p in predictions], 
                               key=lambda x: (x[0], x[1], x[2]))

        matched_preds = set()
        matched_refs = set()

        # Matching avec meilleur score de manière déterministe
        for p_idx, (p_start, p_end, p_label) in enumerate(pred_entities):
            best_match = None
            best_score = float('inf')
            
            for t_idx, (t_start, t_end, t_label) in enumerate(true_entities):
                if p_label == t_label and (t_start, t_end, t_label) not in matched_refs:
                    position_diff = abs(p_start - t_start) + abs(p_end - t_end)
                    
                    # Vérifier le chevauchement effectif
                    has_overlap = max(p_start, t_start) <= min(p_end, t_end)
                    
                    if has_overlap and position_diff <= tolerance and position_diff < best_score:
                        best_score = position_diff
                        best_match = (t_idx, t_start, t_end, t_label)

            if best_match:
                t_idx, t_start, t_end, t_label = best_match
                tp += 1
                tp_i += 1
                class_metrics[p_label]['tp'] += 1
                matched_preds.add((p_start, p_end, p_label))
                matched_refs.add((t_start, t_end, t_label))
            else:
                fp += 1
                fp_i += 1
                if p_label in class_metrics:
                    class_metrics[p_label]['fp'] += 1

        for t_start, t_end, t_label in true_entities:
            if (t_start, t_end, t_label) not in matched_refs:
                fn += 1
                fn_i += 1
                if t_label in class_metrics:
                    class_metrics[t_label]['fn'] += 1

        prec_i = tp_i / (tp_i + fp_i) if (tp_i + fp_i) > 0 else 0
        recall_i = tp_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else 0
        f1_i = 2 * prec_i * recall_i / (prec_i + recall_i) if (prec_i + recall_i) > 0 else 0
        f1_by_doc.append(f1_i)

    # Calcul des métriques comme avant...
    micro_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    micro_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    macro_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'count': 0}
    class_results = {}

    for cls in classes:
        cls_tp = class_metrics[cls]['tp']
        cls_fp = class_metrics[cls]['fp']
        cls_fn = class_metrics[cls]['fn']

        if cls_tp + cls_fn > 0:
            precision = cls_tp / (cls_tp + cls_fp) if (cls_tp + cls_fp) > 0 else 0
            recall = cls_tp / (cls_tp + cls_fn) if (cls_tp + cls_fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            macro_metrics['precision'] += precision
            macro_metrics['recall'] += recall
            macro_metrics['f1'] += f1
            macro_metrics['count'] += 1

            class_results[cls] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': cls_tp + cls_fn,
                'tp': cls_tp, 'fp': cls_fp, 'fn': cls_fn
            }
        else:
            class_results[cls] = {
                'precision': 0, 'recall': 0, 'f1': 0, 'support': 0,
                'tp': 0, 'fp': cls_fp, 'fn': 0
            }

    if macro_metrics['count'] > 0:
        for key in ['precision', 'recall', 'f1']:
            macro_metrics[key] /= macro_metrics['count']

    # Ajouter le mode de segmentation et statistiques supplémentaires
    segmentation_name = "Chunking (no overlap)" if segmentation_mode == 0 else f"Sliding Window ({overlap_ratio*100:.0f}% overlap)"
    
    results = {
        'micro': {'precision': micro_precision, 'recall': micro_recall, 'f1': micro_f1, 
                 'support': tp + fn, 'tp': tp, 'fp': fp, 'fn': fn},
        'macro': macro_metrics,
        'f1_by_doc' : f1_by_doc,
        'by_class': class_results,
        'time': {'total': total_time, 'avg': total_time / len(prepared_data) if prepared_data else 0},
        'segmentation': {'mode': segmentation_mode, 'name': segmentation_name},
        'stats': {
            'total_predictions': total_predictions,
            'total_true_entities': total_true_entities,
            'total_examples': len(prepared_data)
        }
    }

    # results['f1_by_doc'] = f1_by_doc

    if verbose:
        print(f"\n=== ÉVALUATION NER: {segmentation_name} ===")
        print(f"Total exemples: {len(prepared_data)}")
        print(f"Total entités prédites: {total_predictions}")
        print(f"Total entités réelles: {total_true_entities}")
        
        print("\n=== MICRO-AVERAGE ===")
        print(f"Précision: {micro_precision:.4f}")
        print(f"Rappel: {micro_recall:.4f}")
        print(f"F1: {micro_f1:.4f}")
        print(f"TP: {tp}, FP: {fp}, FN: {fn}, Support: {tp + fn}")

        print("\n=== MACRO-AVERAGE ===")
        print(f"Précision: {macro_metrics['precision']:.4f}")
        print(f"Rappel: {macro_metrics['recall']:.4f}")
        print(f"F1: {macro_metrics['f1']:.4f}")

        print("\n=== PAR CLASSE ===")
        for cls, m in sorted(class_results.items(), key=lambda x: x[1]['f1'], reverse=True):
            if m['support'] > 0:
                print(f"{cls:25} - P: {m['precision']:.4f}, R: {m['recall']:.4f}, F1: {m['f1']:.4f}, TP: {m['tp']}, FP: {m['fp']}, FN: {m['fn']}, Sup: {m['support']}")

    return results

def check_max_token_count(data):

    if not data:
        return {"max_tokens": 0, "avg_tokens": 0, "token_counts": [], "texts_exceeding": {}}
    
    token_counts = []
    exceeding_counts = {256: 0, 384: 0, 512: 0, 768: 0, 1024: 0}
    
    for item in data:
        # Obtenir directement la liste de tokens
        tokens = item.get('tokenized_text', [])
        token_count = len(tokens)
        token_counts.append(token_count)
        
        # Compter les textes dépassant différents seuils
        for threshold in exceeding_counts.keys():
            if token_count > threshold:
                exceeding_counts[threshold] += 1
    
    max_tokens = max(token_counts) if token_counts else 0
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    
    # Calculer des percentiles pour voir la distribution
    sorted_counts = sorted(token_counts)
    percentiles = {
        "50th": sorted_counts[len(sorted_counts) // 2] if sorted_counts else 0,
        "75th": sorted_counts[int(len(sorted_counts) * 0.75)] if sorted_counts else 0,
        "90th": sorted_counts[int(len(sorted_counts) * 0.9)] if sorted_counts else 0,
        "95th": sorted_counts[int(len(sorted_counts) * 0.95)] if sorted_counts else 0
    }
    
    # Identifier les documents avec le plus de tokens
    top_documents = []
    if token_counts:
        # Trier les documents par nombre de tokens (décroissant)
        sorted_docs = sorted(enumerate(data), key=lambda x: len(x[1].get('tokenized_text', [])), reverse=True)
        # Prendre les 5 premiers documents
        for i, doc in sorted_docs[:5]:
            top_documents.append({
                "index": i,
                "token_count": len(doc.get('tokenized_text', [])),
                "first_tokens": ' '.join(doc.get('tokenized_text', [])[:10]) + "..." if len(doc.get('tokenized_text', [])) > 10 else ' '.join(doc.get('tokenized_text', [])),
                "entity_count": len(doc.get('ner', []))
            })
    
    return {
        "max_tokens": max_tokens,
        "avg_tokens": avg_tokens,
        "total_documents": len(data),
        "percentiles": percentiles,
        "token_counts": token_counts,
        "texts_exceeding": exceeding_counts,
        "top_documents": top_documents
    }


def segmenter_donnees_ner(data, max_tokens=512, stride=50):
    """
    Segmente les données NER en gardant uniquement les segments qui contiennent au moins une entité.

    Args:
        data: liste de dicts avec 'tokenized_text' et 'ner'
        max_tokens: taille maximale du segment
        stride: recouvrement entre les segments

    Returns:
        Liste de segments avec 'tokenized_text', 'ner' et 'original_start_idx'
    """
    donnees_segmentees = []

    for exemple in data:
        tokens = exemple['tokenized_text']
        spans = exemple['ner']

        if len(tokens) <= max_tokens:
            if spans:  # On garde seulement si une entité existe
                donnees_segmentees.append(exemple)
            continue

        for start_idx in range(0, len(tokens), max_tokens - stride):
            end_idx = min(start_idx + max_tokens, len(tokens))
            segment_tokens = tokens[start_idx:end_idx]

            segment_spans = []
            for span_start, span_end, span_type in spans:
                if span_end >= start_idx and span_start <= end_idx - 1:
                    adjusted_start = max(0, span_start - start_idx)
                    adjusted_end = min(span_end - start_idx, end_idx - start_idx - 1)

                    if adjusted_start <= adjusted_end:
                        segment_spans.append([adjusted_start, adjusted_end, span_type])

            if segment_spans:  # Garder uniquement les segments qui ont des entités
                segment_data = {
                    'tokenized_text': segment_tokens,
                    'ner': segment_spans,
                    'original_start_idx': start_idx
                }
                donnees_segmentees.append(segment_data)

            if end_idx == len(tokens):
                break

    return donnees_segmentees


def label_studio_to_gliner(data):
    """
    Convertit un export Label Studio (format réel) en format :
    [{'tokenized_text': [...], 'ner': [[start_idx, end_idx, label], ...]}]

    Utilise une tokenisation qui sépare la ponctuation et conserve les apostrophes seules comme tokens distincts.
    """
    result = []

    for item in data:
        text = item['data'].get('text')
        if not text:
            continue

        annotations = item.get('annotations', [])
        if not annotations:
            continue

        # Tokenisation : mots + apostrophes + ponctuation séparée, y compris guillemets français
        tokens = re.findall(r"\w+|''|'|[.,!?;:\"«»()\-]", text)

        # Calculer les positions des tokens dans le texte
        token_spans = []
        offset = 0
        for token in tokens:
            match = re.search(re.escape(token), text[offset:])
            if match:
                start = offset + match.start()
                end = start + len(token)
                token_spans.append((start, end))
                offset = end

        ner = []
        for ann in annotations[0]['result']:
            value = ann['value']
            ent_start = value['start']
            ent_end = value['end']
            label = value['labels'][0]

            token_start = token_end = None
            for idx, (s, e) in enumerate(token_spans):
                if token_start is None and s <= ent_start < e:
                    token_start = idx
                if token_end is None and s < ent_end <= e:
                    token_end = idx
                if s >= ent_end:
                    break

            if token_start is not None and token_end is None:
                token_end = token_start

            if token_start is not None and token_end is not None:
                ner.append([token_start, token_end, label])

        result.append({'tokenized_text': tokens, 'ner': ner})

    return result

def analyze_entities_in_data(filtered_data):
    """Analyse les entités présentes dans chaque article"""
    article_entities = []
    
    for article in filtered_data:
        entities_by_type = defaultdict(set)
        
        # Parcourir les annotations
        for annotation in article.get('annotations', []):
            for result in annotation.get('result', []):
                if result.get('type') == 'labels':
                    entity_type = result['value']['labels'][0]
                    entity_text = result['value']['text']
                    entities_by_type[entity_type].add(entity_text.lower().strip())
        
        article_entities.append({
            'article_id': article.get('id', len(article_entities)),
            'article_data': article,
            'entities_by_type': dict(entities_by_type),
            'entity_counts': {k: len(v) for k, v in entities_by_type.items()}
        })
    
    return article_entities

def create_balanced_split(filtered_data, test_size=30, min_entities_per_type=10):
    """
    Crée un split équilibré garantissant min_entities_per_type entités 
    de chaque type dans le test set
    """
    
    print(f"Objectif: {test_size} articles test avec au moins {min_entities_per_type} entités par type")
    
    # Analyser les entités dans chaque article
    article_entities = analyze_entities_in_data(filtered_data)
    
    # Compter les entités globales par type
    global_entities_by_type = defaultdict(set)
    for article in article_entities:
        for entity_type, entities in article['entities_by_type'].items():
            global_entities_by_type[entity_type].update(entities)
    
    print("\n=== ANALYSE GLOBALE ===")
    print(f"Nombre total d'articles: {len(article_entities)}")
    for entity_type, entities in global_entities_by_type.items():
        print(f"{entity_type}: {len(entities)} entités uniques")
    
    # Vérifier la faisabilité
    entity_types = list(global_entities_by_type.keys())
    for entity_type in entity_types:
        if len(global_entities_by_type[entity_type]) < min_entities_per_type:
            print(f"⚠️  ATTENTION: {entity_type} n'a que {len(global_entities_by_type[entity_type])} entités uniques!")
    
    # Algorithme de sélection équilibrée
    test_articles = []
    test_entities_by_type = defaultdict(set)
    remaining_articles = article_entities.copy()
    
    print("\n=== SÉLECTION DES ARTICLES TEST ===")
    
    # Phase 1: Sélectionner des articles riches pour chaque type d'entité
    for entity_type in entity_types:
        print(f"\nPhase 1 - Type: {entity_type}")
        
        # Trier les articles par nombre d'entités de ce type
        candidates = [a for a in remaining_articles if entity_type in a['entities_by_type']]
        candidates.sort(key=lambda x: x['entity_counts'].get(entity_type, 0), reverse=True)
        
        # Prendre les meilleurs candidats jusqu'à avoir assez d'entités
        entities_needed = max(0, min_entities_per_type - len(test_entities_by_type[entity_type]))
        
        for article in candidates:
            if len(test_articles) >= test_size:
                break
            if article in test_articles:
                continue
            if entities_needed <= 0:
                break
                
            # Ajouter l'article au test set
            test_articles.append(article)
            remaining_articles.remove(article)
            
            # Mettre à jour le compteur d'entités test
            for etype, entities in article['entities_by_type'].items():
                test_entities_by_type[etype].update(entities)
            
            entities_added = len(article['entities_by_type'].get(entity_type, set()))
            entities_needed -= entities_added
            
            print(f"  Article {article['article_id']}: +{entities_added} entités -> Total {entity_type}: {len(test_entities_by_type[entity_type])}")
    
    # Phase 2: Compléter jusqu'à 30 articles si nécessaire
    print(f"\nPhase 2 - Compléter jusqu'à {test_size} articles")
    while len(test_articles) < test_size and remaining_articles:
        # Choisir un article aléatoire parmi les restants
        article = random.choice(remaining_articles)
        test_articles.append(article)
        remaining_articles.remove(article)
        
        # Mettre à jour les compteurs
        for entity_type, entities in article['entities_by_type'].items():
            test_entities_by_type[entity_type].update(entities)
        
        print(f"  Article {article['article_id']} ajouté (total: {len(test_articles)})")
    
    # Créer les datasets finaux
    test_data = [article['article_data'] for article in test_articles]
    train_data = [article['article_data'] for article in remaining_articles]
    
    # Statistiques finales
    print("\n=== RÉSULTATS FINAUX ===")
    print(f"Articles d'entraînement: {len(train_data)}")
    print(f"Articles de test: {len(test_data)}")
    
    print("\nEntités dans le test set:")
    for entity_type in entity_types:
        count = len(test_entities_by_type[entity_type])
        status = "✅" if count >= min_entities_per_type else "❌"
        print(f"  {entity_type}: {count} entités {status}")
    
    return train_data, test_data, test_entities_by_type

# Utilisation
def split_ner_data(filtered_data, test_size, min_entities_per_type):
    """Fonction principale pour splitter les données NER"""
    
    print("🔄 Création du split train/test équilibré...")
    
    # Fixer la seed pour la reproductibilité
    random.seed(42)
    
    # Créer le split
    train_data, test_data, test_stats = create_balanced_split(
        filtered_data, 
        test_size= test_size, 
        min_entities_per_type= min_entities_per_type
    )
    
    # Sauvegarder les résultats
    print("\n💾 Sauvegarde des datasets...")
    
    with open('train_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open('test_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Datasets sauvegardés:")
    print(f"   - train_dataset.json: {len(train_data)} articles")
    print(f"   - test_dataset.json: {len(test_data)} articles")
    
    return train_data, test_data


def remove_text_duplicates(annotations):

    seen_texts = {}
    unique_annotations = []
    
    for item in annotations:

        if "data" in item and "text" in item["data"]:
            text = item["data"]["text"]
            
            if text not in seen_texts:
                seen_texts[text] = True
                unique_annotations.append(item)
                
    return unique_annotations


def filter_and_lowercase_entity_types(data, entity_types_to_remove):

    if isinstance(entity_types_to_remove, str):
        entity_types_to_remove = [entity_types_to_remove]
    
    # Créer une copie profonde des données pour éviter les modifications accidentelles
    import copy
    filtered_data = copy.deepcopy(data)
    
    # Parcourir chaque élément des données
    for item in filtered_data:
        # Si nous avons déjà un format filtré (structure tuple)
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], dict) and 'entities' in item[1]:
            text, annotations = item
            # Filtrer les entités
            filtered_entities = []
            for start, end, label in annotations['entities']:
                if label not in entity_types_to_remove:
                    # Convertir en minuscules
                    filtered_entities.append((start, end, label.lower()))
            # Mettre à jour les annotations
            annotations['entities'] = filtered_entities
        
        # Format Label Studio original
        elif 'annotations' in item and item['annotations']:
            for i, annotation_set in enumerate(item['annotations']):
                filtered_results = []
                
                for result in annotation_set.get('result', []):
                    if 'value' in result and 'labels' in result['value']:
                        # Vérifier si aucune des étiquettes n'est dans la liste à supprimer
                        labels = result['value']['labels']
                        if not any(label in entity_types_to_remove for label in labels):
                            # Convertir toutes les étiquettes en minuscules
                            result['value']['labels'] = [label.lower() for label in labels]
                            filtered_results.append(result)
                    else:
                        # Conserver les autres types d'annotations
                        filtered_results.append(result)
                
                # Mettre à jour les résultats filtrés
                item['annotations'][i]['result'] = filtered_results
    
    return filtered_data

def augmenter_donnees_ner(data_label_studio, type_entite, replacements, max_augmentations=3):

    import random
    import copy
    import uuid
    
    resultats_augmentes = []
    
    # Parcourir chaque exemple du jeu de données
    for exemple in data_label_studio:
        # Vérifier la présence des données nécessaires
        if 'data' not in exemple or 'text' not in exemple['data']:
            continue
            
        texte_original = exemple['data']['text']
        
        # Chercher les entités du type ciblé
        entites_cibles = []
        
        # Vérifier si l'exemple a des annotations
        if 'annotations' in exemple and exemple['annotations']:
            # Parcourir chaque annotation
            for annotation_idx, annotation in enumerate(exemple['annotations']):
                if 'result' not in annotation:
                    continue
                    
                # Parcourir chaque entité dans les résultats
                for entity_idx, entity in enumerate(annotation['result']):
                    # Vérifier si l'entité a le bon type
                    if ('value' in entity and 'labels' in entity['value'] and 
                        type_entite.lower() in [label.lower() for label in entity['value']['labels']]):
                        
                        # Stocker l'entité avec des infos sur sa localisation
                        entity_copy = copy.deepcopy(entity)
                        entity_copy['_annotation_idx'] = annotation_idx
                        entity_copy['_entity_idx'] = entity_idx
                        entites_cibles.append(entity_copy)
        
        # Si aucune entité cible, passer à l'exemple suivant
        if not entites_cibles:
            continue
        
        # Créer des versions augmentées
        for _ in range(min(max_augmentations, len(replacements))):
            # Copier l'exemple original
            exemple_augmente = copy.deepcopy(exemple)
            exemple_augmente['id'] = exemple['id'] if isinstance(exemple['id'], int) else int(exemple['id'])
            
            # Générer un nouveau texte en remplaçant les entités
            nouveau_texte = texte_original
            decalage = 0  # Pour suivre le décalage dans les positions
            
            # Trier les entités par position de début (ordre décroissant pour éviter les problèmes de décalage)
            entites_triees = sorted(entites_cibles, key=lambda e: e['value']['start'], reverse=True)
            
            for entite in entites_triees:
                # Choisir un texte de remplacement aléatoire
                nouveau_texte_entite = random.choice(replacements)
                
                # Extraire les positions de l'entité
                start = entite['value']['start']
                end = entite['value']['end']
                
                # Calculer la différence de longueur
                diff_longueur = len(nouveau_texte_entite) - (end - start)
                
                # Remplacer l'entité dans le texte
                nouveau_texte = nouveau_texte[:start + decalage] + nouveau_texte_entite + nouveau_texte[end + decalage:]
                
                # Récupérer la localisation de l'entité
                annotation_idx = entite.get('_annotation_idx', 0)
                entity_idx = entite.get('_entity_idx', 0)
                entity_id = entite.get('id', None)
                
                # Mettre à jour l'entité dans l'exemple augmenté
                if ('annotations' in exemple_augmente and 
                    annotation_idx < len(exemple_augmente['annotations']) and
                    'result' in exemple_augmente['annotations'][annotation_idx]):
                    
                    annotation = exemple_augmente['annotations'][annotation_idx]
                    
                    # Parcourir les entités de l'annotation
                    for idx, entity in enumerate(annotation['result']):
                        if idx == entity_idx or (entity_id and 'id' in entity and entity['id'] == entity_id):
                            # Mettre à jour cette entité
                            entity['value']['text'] = nouveau_texte_entite
                            entity['value']['end'] = entity['value']['start'] + len(nouveau_texte_entite)
                        elif 'value' in entity and entity['value']['start'] > start + decalage:
                            # Ajuster les positions des entités qui suivent
                            entity['value']['start'] += diff_longueur
                            entity['value']['end'] += diff_longueur
                
                # Mettre à jour le décalage pour la prochaine entité
                decalage += diff_longueur
            
            # Mettre à jour le texte
            exemple_augmente['data']['text'] = nouveau_texte
            
            # Supprimer les métadonnées temporaires
            if 'annotations' in exemple_augmente:
                for annotation in exemple_augmente['annotations']:
                    if 'result' in annotation:
                        for entity in annotation['result']:
                            for key in ['_annotation_idx', '_entity_idx']:
                                if key in entity:
                                    del entity[key]
            
            # Générer un nouvel identifiant unique pour l'exemple augmenté
            exemple_augmente['id'] = int(str(exemple_augmente['id']) + str(random.randint(1000, 9999)))
            
            resultats_augmentes.append(exemple_augmente)
    
    return resultats_augmentes


def augmenter_donnees_multitypes(data_label_studio, entities_lists, max_augmentations_par_type=3):

    toutes_donnees_augmentees = []
    
    # Appliquer l'augmentation pour chaque type d'entité
    for type_entite, replacements in entities_lists.items():
        print(f"Augmentation pour le type: {type_entite}")
        
        try:
            # Appliquer la fonction d'augmentation pour ce type
            donnees_augmentees = augmenter_donnees_ner(
                data_label_studio=data_label_studio, 
                type_entite=type_entite, 
                replacements=replacements,
                max_augmentations=max_augmentations_par_type
            )
            
            # Ajouter les résultats à la liste globale
            toutes_donnees_augmentees.extend(donnees_augmentees)
            print(f"  Nombre d'exemples générés: {len(donnees_augmentees)}")
            
        except Exception as e:
            print(f"  Erreur lors de l'augmentation pour le type '{type_entite}': {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Total d'exemples augmentés: {len(toutes_donnees_augmentees)}")
    return toutes_donnees_augmentees

def analyze_labelstudio_annotations_unique_with_list(annotations):
    # Dictionnaire pour stocker les entités uniques par label
    label_to_unique_entities = defaultdict(set)

    for annotation in annotations:
        results = annotation.get('annotations', [])
        for result in results:
            for item in result.get('result', []):
                if item['type'] == 'labels':
                    labels = item['value']['labels']
                    text = item['value']['text'].strip().lower()  # Normalisation
                    for label in labels:
                        label_to_unique_entities[label].add(text)

    # Compter et afficher
    print("📊 Statistiques d'entités uniques par label :")
    total_unique_entities = 0
    for label, entities in label_to_unique_entities.items():
        count = len(entities)
        total_unique_entities += count
        print(f"\n🔹 {label} ({count} entités uniques) :")
        for e in sorted(entities):
            print(f"  - {e}")

    print(f"\n📈 Nombre total d'entités uniques : {total_unique_entities}")

    # 📊 Diagramme
    entity_counter = {label: len(entities) for label, entities in label_to_unique_entities.items()}
    plt.figure(figsize=(10, 6))
    plt.bar(entity_counter.keys(), entity_counter.values(), color='mediumseagreen')
    plt.title("Nombre d'entités uniques par label")
    plt.xlabel("Label")
    plt.ylabel("Nombre d'entités uniques")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return label_to_unique_entities  # utile pour réutiliser la structure ailleurs


def analyze_labelstudio_annotations(annotations):

    entity_counter = Counter()

    for annotation in annotations:
        results = annotation.get('annotations', [])
        for result in results:
            for item in result.get('result', []):
                if item['type'] == 'labels':
                    labels = item['value']['labels']
                    for label in labels:
                        entity_counter[label] += 1

    total_entities = sum(entity_counter.values())

    print("📊 Statistiques d'annotations :")
    print(f"Nombre total d'entités annotées : {total_entities}")
    print("Répartition par label :")
    for label, count in entity_counter.items():
        print(f"- {label} : {count}")

    # 📈 Tracer un graphique en barres
    plt.figure(figsize=(10, 6))
    plt.bar(entity_counter.keys(), entity_counter.values(), color='skyblue')
    plt.title('Nombre d\'entités annotées par label')
    plt.xlabel('Label')
    plt.ylabel('Nombre d\'entités')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def convert_label_studio_to_examples(annotations, start_index=0, end_index=None, max_examples=None):

    examples = []

    selected_annotations = annotations[start_index:end_index]

    max_to_process = len(selected_annotations)
    if max_examples is not None:
        max_to_process = min(max_to_process, max_examples)

    for i, item in enumerate(selected_annotations):
        if i >= max_to_process:
            break
            
        if "annotations" in item and "data" in item:
            text = item["data"].get("text", "")
            entities = []
            
            for annotation in item["annotations"]:
                if "result" in annotation:
                    for result in annotation["result"]:
                        if "value" in result and "labels" in result["value"]:
                            entity_text = result["value"].get("text", "")
                            label = result["value"]["labels"][0] 
                            entities.append([entity_text, label])
            
            if text and entities:
                examples.append({
                    "text": text,
                    "entities": entities
                })
    
    return examples

def create_few_shot_prompt(examples):
    examples_text = ""
    
    for i, example in enumerate(examples):
        examples_text += f"\n\nEXEMPLE {i+1}:\nTexte: {example['text']}\n\nEntités identifiées:\n```json\n{{\n  \"entities\": {json.dumps(example['entities'], ensure_ascii=False, indent=2)}\n}}\n```"
    
    return examples_text

import json
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyser_entites(document_content):

    types_attendus = {
        "Magasin de producteurs": "1",
        "Réseau de magasins": "2",
        "Point de vente": "3",
        "Producteur/Artisan": "4",
        "Produit": "5",
        "Organisation": "6",
        "Nombre de magasins": "7",
        "Information temporelle": "8",
        "Lieu": "9",
        "Déclencheur": "0",
        "Nombre de producteurs": "q",
        "Autres": "w"
    }
    
    compteur_types = {}
    entites_par_type = {}

    try:
        docs = eval(document_content) if isinstance(document_content, str) else document_content
        docs = [docs] if not isinstance(docs, list) else docs

        for doc in docs:
            for entity in doc.get('entities', []):
                if len(entity) >= 2:
                    texte, type_entite = entity[0], entity[1]
                    compteur_types[type_entite] = compteur_types.get(type_entite, 0) + 1
                    entites_par_type.setdefault(type_entite, [])
                    if texte not in entites_par_type[type_entite]:
                        entites_par_type[type_entite].append(texte)
        
        print("=== OCCURRENCES PAR TYPE D'ENTITÉ ===")
        for type_entite, count in sorted(compteur_types.items(), key=lambda x: x[1], reverse=True):
            print(f"{type_entite}: {count} occurrences")
            print(f"  Exemples: {', '.join(entites_par_type[type_entite][:3])}")
            if len(entites_par_type[type_entite]) > 3:
                print(f"  ... et {len(entites_par_type[type_entite])-3} autres")

        types_presents = set(compteur_types.keys())
        types_attendus_set = set(types_attendus.keys())
        
        types_absents = types_attendus_set - types_presents
        types_inattendus = types_presents - types_attendus_set
        
        print("\n=== ANALYSE DE COUVERTURE DES TYPES D'ENTITÉS ===")
        print(f"Types d'entités présents: {len(types_presents)}/{len(types_attendus)}")
        
        if types_absents:
            print("\nTypes d'entités absents:")
            for type_absent in sorted(types_absents):
                print(f"  - {type_absent} (code: {types_attendus[type_absent]})")
        
        if types_inattendus:
            print("\nTypes d'entités non attendus trouvés:")
            for type_inattendu in sorted(types_inattendus):
                print(f"  - {type_inattendu}: {compteur_types[type_inattendu]} occurrences")
        
    except Exception as e:
        print(f"Erreur: {e}")

def annoter_label_studio(texte, model_output):

    PRIORITY = ["Magasin de producteurs", "Réseau de magasins", "Point de vente", "Organisation"]
    VALID_LABELS = ["Magasin de producteurs", "Réseau de magasins", "Point de vente", "Organisation", 
                    "Nombre de producteurs", "Producteur/Artisan", "Produit", "Information temporelle", "Lieu", "Déclencheur"]

    entities = []
    entity_pattern = r'\["([^"]+)", "([^"]+)"\]'
    
    try:
        if '```json' in model_output:
            json_match = re.search(r'```json\s*(.*?)(\s*```|\Z)', model_output, re.DOTALL)
            if json_match:
                entity_matches = re.findall(entity_pattern, json_match.group(1))
                entities = [list(match) for match in entity_matches]
        elif '"entities":' in model_output:
            data = json.loads(re.search(r'(\{.*\})', model_output, re.DOTALL).group(1))
            if "entities" in data:
                entities = data["entities"]
        else:
            entity_matches = re.findall(entity_pattern, model_output)
            entities = [list(match) for match in entity_matches]
    except:

        entities = re.findall(entity_pattern, model_output)
        entities = [list(match) for match in entities]
    
    results = []
    seen_entities = {}
    
    for entity in entities:
        if isinstance(entity, list) and len(entity) >= 2:
            text, label = entity[0].strip(), entity[1].strip()
            if label in VALID_LABELS:
                seen_entities.setdefault(text, set()).add(label)
    
    for entity_text, labels in seen_entities.items():
        label = next((l for l in PRIORITY if l in labels), next(iter(labels)))
        
        spans = []
        for match in re.finditer(re.escape(entity_text), texte):
            spans.append((match.start(), match.end()))
        
        if not spans:
            for match in re.finditer(re.escape(entity_text), texte, re.IGNORECASE):
                spans.append((match.start(), match.end()))
                
        for start, end in spans:
            results.append({
                "id": f"result_{len(results)+1}",
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "value": {
                    "start": start,
                    "end": end,
                    "text": texte[start:end],
                    "labels": [label]
                }
            })
    
    return json.dumps([{"data": {"text": texte}, "annotations": [{"result": results}]}], indent=2, ensure_ascii=False)

def parse_segments(raw_segments):
    if not raw_segments or raw_segments == '[]':
        return []
    
    try:
        return json.loads(raw_segments)
    except json.JSONDecodeError:
        if '\'' in raw_segments:
            segments = re.findall(r'\'([^\']+)\'', raw_segments)
        elif '"' in raw_segments:
            segments = re.findall(r'"([^"]+)"', raw_segments)
        else:
            return []
        
        return segments if segments else []

def convertir_en_annotations_segments(annotations, tokenizer, max_tokens=512, stride=153):

    resultats = []

    for article in annotations:
        segments = parse_segments(article['data'].get('Segments', '[]'))
        texte_complet = article['data'].get('text', '')
        annotations_par_segment = []

        for segment in segments:
            pos_segment = texte_complet.find(segment)
            if pos_segment == -1:
                continue  

            sous_segments = split_text_into_sliding_windows(segment, tokenizer, max_tokens=max_tokens, stride=stride)

            for sous_segment in sous_segments:
                sous_pos = texte_complet.find(sous_segment, pos_segment)  # Recherche locale à partir du segment
                if sous_pos == -1:
                    continue

                annotation_segment = {
                    "data": {"text": sous_segment},
                    "annotations": [{"result": []}]
                }

                for entite in article.get('annotations', [{}])[0].get('result', []):
                    start_entite = entite['value']['start']
                    end_entite = entite['value']['end']

                    if start_entite >= sous_pos and end_entite <= sous_pos + len(sous_segment):
                        nouvelle_entite = entite.copy()
                        nouvelle_entite['value'] = entite['value'].copy()
                        nouvelle_entite['value']['start'] = start_entite - sous_pos
                        nouvelle_entite['value']['end'] = end_entite - sous_pos
                        annotation_segment['annotations'][0]['result'].append(nouvelle_entite)

                annotations_par_segment.append(annotation_segment)

        resultats.append(annotations_par_segment)

    return resultats

def convertir_label_studio_vers_entites(annotations_segmentees):

    resultat = {"classes": set(), "annotations": []}
    
    for article_segments in annotations_segmentees:
        article_annotations = []
        
        for segment_annotation in article_segments:
            segment_text = segment_annotation["data"]["text"]
            entities = []
            
            for annotation in segment_annotation["annotations"][0]["result"]:
                start = annotation["value"]["start"]
                end = annotation["value"]["end"]
                label = annotation["value"]["labels"][0]

                entities.append([start, end, label])

                resultat["classes"].add(label)
            
            article_annotations.append([segment_text, {"entities": entities}])

        resultat["annotations"].extend(article_annotations)
    
    resultat["classes"] = sorted(list(resultat["classes"]))
    
    return resultat

def reconstruire_annotations(segment_annotations, texte_complet):

    annotation_unifiee = {"data": {"text": texte_complet}, "annotations": [{"result": []}]}
    result = annotation_unifiee["annotations"][0]["result"]
    id_counter = 1
    
    entites_uniques = {}
    
    for segment_ann in segment_annotations:
        for ann in segment_ann.get("annotations", [])[0].get("result", []):
            entity_text = ann["value"]["text"]
            entity_type = ann["value"]["labels"][0]
            
            start_search = 0
            while True:
                pos = texte_complet.find(entity_text, start_search)
                if pos == -1:
                    break
                
                result.append({
                    "id": f"result_{id_counter}",
                    "from_name": "label",
                    "to_name": "text",
                    "type": "labels",
                    "value": {
                        "start": pos,
                        "end": pos + len(entity_text),
                        "text": entity_text,
                        "labels": [entity_type]
                    }
                })
                id_counter += 1
                
                start_search = pos + 1
    
    seen_positions = set()
    unique_results = []
    
    for ann in result:
        position = (ann["value"]["start"], ann["value"]["end"])
        if position not in seen_positions:
            seen_positions.add(position)
            unique_results.append(ann)
    
    annotation_unifiee["annotations"][0]["result"] = unique_results
    return annotation_unifiee


def compute_ner_metrics(pred_annotation, ref_annotation):
    from collections import Counter, defaultdict
    
    def extract_entities(annotation):
        entities = set()
        if isinstance(annotation, list):
            for item in annotation:
                if isinstance(item, dict) and "annotations" in item:
                    for ann in item["annotations"][0]["result"]:
                        start = ann["value"]["start"]
                        end = ann["value"]["end"]
                        label = ann["value"]["labels"][0]
                        entities.add((start, end, label))
        elif isinstance(annotation, dict):
            for ann in annotation.get("annotations", [])[0]["result"]:
                start = ann["value"]["start"]
                end = ann["value"]["end"]
                label = ann["value"]["labels"][0]
                entities.add((start, end, label))
        return entities
    
    tp = 0
    fp = 0
    fn = 0
    
    class_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    for pred, ref in zip(pred_annotation, ref_annotation):
        pred_ents = extract_entities(pred)
        ref_ents = extract_entities(ref)
        
        # Métriques globales
        tp += len(pred_ents & ref_ents)
        fp += len(pred_ents - ref_ents)
        fn += len(ref_ents - pred_ents)
        
        # Métriques par classe
        for start, end, label in (pred_ents & ref_ents):
            class_metrics[label]["tp"] += 1
            
        for start, end, label in (pred_ents - ref_ents):
            class_metrics[label]["fp"] += 1
            
        for start, end, label in (ref_ents - pred_ents):
            class_metrics[label]["fn"] += 1
    
    # Calcul métriques globales
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calcul métriques par classe
    per_class = {}
    for label, counts in class_metrics.items():
        cls_precision = counts["tp"] / (counts["tp"] + counts["fp"]) if (counts["tp"] + counts["fp"]) > 0 else 0
        cls_recall = counts["tp"] / (counts["tp"] + counts["fn"]) if (counts["tp"] + counts["fn"]) > 0 else 0
        cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall) if (cls_precision + cls_recall) > 0 else 0
        
        per_class[label] = {
            "true_positives": counts["tp"],
            "false_positives": counts["fp"],
            "false_negatives": counts["fn"],
            "precision": round(cls_precision, 4),
            "recall": round(cls_recall, 4),
            "f1_score": round(cls_f1, 4),
            "support": counts["tp"] + counts["fn"]
        }
    
    return {
        "overall": {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        },
        "per_class": per_class
    }