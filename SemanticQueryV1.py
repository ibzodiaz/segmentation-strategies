import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
from tqdm import tqdm
from datetime import datetime
import logging

@dataclass
class SearchConfig:
    threshold: float = 0.2
    batch_size: int = 8
    max_sequence_length: int = 512

class SentenceDataset(Dataset):
    def __init__(self, sentences: List[str]):
        self.sentences = sentences

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> str:
        return self.sentences[idx]

class SemanticSearchEngine:
    def __init__(self, model_name: str, config: SearchConfig):
        """Initialise le moteur de recherche."""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._segments_embeddings_cache = {}  # Cache uniquement pour les segments
        
        # Initialisation du modèle SBERT
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
        except Exception as e:
            raise RuntimeError(f"Erreur lors de l'initialisation du modèle: {e}")
        
        # Configuration du logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def preprocess_text(self, text: str) -> str:
        """Prétraite le texte pour la recherche."""
        # Normalisation des caractères spéciaux
        text = re.sub(r'[«»"""]', '"', text)
        text = re.sub(r'[''‛]', "'", text)
        
        # Normalisation des espaces et ponctuation
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s*([.,!?;:])\s*', r'\1 ', text)
        
        # Normalisation des nombres
        text = re.sub(r'(\d{1,3})\s+(\d{3})', r'\1\2', text)
        
        return text.strip()

    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Calcule les embeddings des textes avec SBERT."""
        dataset = SentenceDataset(texts)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        embeddings = []

        for batch in tqdm(dataloader, desc="Calcul des embeddings"):
            # Avec SBERT, le calcul des embeddings est simplifié
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
            embeddings.append(batch_embeddings)

        return np.concatenate(embeddings, axis=0)
    
    def _get_cached_segments_embeddings(self, segments: List[str]) -> np.ndarray:
        """
        Récupère les embeddings des segments depuis le cache ou les calcule si nécessaire
        """
        # Si le cache est vide, calculer tous les embeddings
        if not self._segments_embeddings_cache:
            embeddings = self.compute_embeddings(segments)
            for segment, embedding in zip(segments, embeddings):
                self._segments_embeddings_cache[segment] = embedding
            return embeddings

        # Si les segments sont déjà en cache, les retourner dans le bon ordre
        return np.array([self._segments_embeddings_cache[segment] for segment in segments])

    def search(self, query: str, documents: List[List[str]]) -> List[Dict[str, Any]]:
    
        # Aplatir les documents en liste de segments avec référence au document d'origine
        flat_segments = []
        doc_references = []
        
        for doc_idx, document in enumerate(documents):
            for seg_idx, segment in enumerate(document):
                flat_segments.append(segment)
                doc_references.append((doc_idx, seg_idx))
        
        self.logger.info(f"Début de la recherche avec {len(flat_segments)} segments issus de {len(documents)} documents")
        
        # Prétraitement
        query = self.preprocess_text(query)
        processed_segments = [self.preprocess_text(seg) for seg in flat_segments]
        
        # Calcul de l'embedding de la requête
        query_embedding = self.compute_embeddings([query])[0]
        query_embedding = query_embedding.reshape(1, -1)
        
        # Récupération des embeddings des segments
        segment_embeddings = self._get_cached_segments_embeddings(processed_segments)
        
        # Normalisation
        query_embedding = normalize(query_embedding)
        segment_embeddings = normalize(segment_embeddings)
        
        results = []
        for idx, (segment, (doc_idx, seg_idx)) in enumerate(zip(processed_segments, doc_references)):
            # Calcul de la similarité sémantique
            semantic_sim = float(cosine_similarity(query_embedding, 
                                                segment_embeddings[idx].reshape(1, -1))[0, 0])
        
            # Application du seuil
            if semantic_sim > self.config.threshold:
                # Utiliser le segment original (non prétraité) pour les résultats
                results.append({
                    "Document_Index": doc_idx,
                    "Segment_Index": seg_idx,
                    "Segment": flat_segments[idx],  # Segment original
                    "Score": semantic_sim
                })
        
        # Trier les résultats par score décroissant
        results = sorted(results, key=lambda x: x["Score"], reverse=True)
        
        self.logger.info(f"Recherche terminée: {len(results)} résultats retournés")
        
        return results