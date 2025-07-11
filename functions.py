import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from termcolor import colored
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


def preprocess_text(text):
    # Mettre le texte en minuscule
    text = text.lower()

    # Normaliser les accents
    #text = normalize_accents(text)

    # Supprimer les contractions du type "l'", "d'", "m'", etc.
    #text = re.sub(r"\b[ldmctjsnqu]'", '', text)
    
    unwanted_tokens = [",", ".", ":", "(", ")", "?", "!", "«", "»","€",'"','%']
    # Créer une expression régulière qui correspond à ces caractères spécifiques
    pattern = f"[{re.escape(''.join(unwanted_tokens))}]"
    # Utiliser re.sub pour remplacer les caractères correspondants par une chaîne vide
    text = re.sub(pattern, '', text)
    
    # Supprimer les espaces superflus
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def documents_preprocessed(documents):
    documents_preprocessed = []
    for doc in documents:
        # Convertir en chaîne de caractères et remplacer les NaN par des chaînes vides
        doc = str(doc) if pd.notna(doc) else ''
        documents_preprocessed.append(preprocess_text(doc))
    return documents_preprocessed

def merge_tokens_and_labels(tokens, labels, prefix_char='▁', 
                             merge_condition=None, 
                             label_merge_strategy=None):
    """
    Fusionne efficacement les tokens et leurs labels correspondants.
    
    Args:
        tokens (list): Liste des tokens
        labels (list): Liste des labels correspondants
        prefix_char (str, optional): Caractère indiquant le début d'un token. Défaut '▁'.
        merge_condition (callable, optional): Fonction de condition personnalisée pour la fusion.
        label_merge_strategy (callable, optional): Fonction de stratégie de fusion des labels.
    
    Returns:
        tuple: (tokens_fusionnes, labels_fusionnes)
    """
    if len(tokens) != len(labels):
        raise ValueError("Les listes de tokens et de labels doivent avoir la même longueur")
    
    # Fonctions de merge par défaut utilisant des lambdas pour plus de concision
    merge_condition = merge_condition or (lambda t, nt: not nt.startswith(prefix_char))
    label_merge_strategy = label_merge_strategy or (lambda cl, nl: nl if cl == nl else nl)
    
    tokens_fusionnes, labels_fusionnes = [], []
    i = 0
    
    while i < len(tokens):
        # Début de fusion si le token commence par le préfixe
        if tokens[i].startswith(prefix_char):
            # Utiliser une liste pour collecter les tokens à fusionner
            merge_group = [tokens[i]]
            label_courant = labels[i]
            
            # Extension du groupe de fusion
            j = i + 1
            while j < len(tokens) and merge_condition(tokens[i], tokens[j]):
                merge_group.append(tokens[j])
                label_courant = label_merge_strategy(label_courant, labels[j])
                j += 1
            
            # Fusion efficace des tokens
            tokens_fusionnes.append(''.join(merge_group))
            labels_fusionnes.append(label_courant)
            
            i = j
        else:
            # Token sans préfixe ajouté tel quel
            tokens_fusionnes.append(tokens[i])
            labels_fusionnes.append(labels[i])
            i += 1
    
    return tokens_fusionnes, labels_fusionnes

def normalize_text(text):
    """
    Normalise le texte en remplaçant plusieurs espaces par un seul et en supprimant les espaces en début et fin.
    """
    # Remplacer plusieurs espaces par un seul
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def analyser_termes_magasin(liste1, liste2):
    """
    Analyse les articles de liste2 pour repérer les termes de liste1.
    Retourne une liste de tuples avec les spans et étiquettes.
    Priorise les termes les plus longs et évite les annotations chevauchantes.
    """
    resultats = []

    # Trier les termes par longueur décroissante pour prioriser les termes les plus longs
    liste1_triee = sorted(liste1, key=lambda x: len(x), reverse=True)

    # Construire une expression régulière unique avec tous les termes, en échappant les caractères spéciaux
    # Utiliser des groupes non capturants (?:...) pour l'alternance
    # Utiliser \b pour les bornes de mots
    pattern_regex = r'\b(?:' + '|'.join(re.escape(terme) for terme in liste1_triee) + r')\b'

    # Compiler l'expression régulière avec l'option IGNORECASE
    pattern = re.compile(pattern_regex, re.IGNORECASE)

    # Parcours des articles dans liste2
    for article in liste2:
        if not isinstance(article, str):  # Vérifie que l'article est bien une chaîne
            continue

        normalized_article = normalize_text(article)
        entities = []

        # Recherche des correspondances dans l'article
        for match in pattern.finditer(normalized_article):
            start, end = match.span()
            matched_text = normalized_article[start:end]
            entities.append((start, end, 'magasin'))
            #print(f"Correspondance trouvée: '{matched_text}' à la position {start}-{end}")

        # Filtrer les correspondances chevauchantes en privilégiant les termes les plus longs
        if entities:
            # Trier les entités par position de début, puis par longueur décroissante
            entities_sorted = sorted(entities, key=lambda x: (x[0], -(x[1]-x[0])))

            entities_non_chevauchantes = []
            dernier_end = -1

            for ent in entities_sorted:
                start, end, label = ent
                if start >= dernier_end:
                    entities_non_chevauchantes.append(ent)
                    dernier_end = end  # Mettre à jour la dernière position de fin

            # Ajout des entités non chevauchantes pour cet article
            if entities_non_chevauchantes:
                resultats.append((normalized_article, {'entities': entities_non_chevauchantes}))

    return resultats

def afficher_annotes(resultats):
    """
    Affiche les articles avec les entités annotées en couleur.
    """
    cp= 0
    for article, data in resultats:
        output = ""
        last_pos = 0
        for start, end, label in sorted(data['entities'], key=lambda x: x[0]):
            output += article[last_pos:start]  # Texte normal
            output += colored(article[start:end], 'red')  # Texte annoté en rouge
            last_pos = end
        output += article[last_pos:]  # Texte restant
        print("\nSegment annoté :")
        print(output)
        cp+=1
    print(f"Nombre de segment : {cp}")

def segment_text(articles, max_tokens=256, fixed_size=True, overlap=False, overlap_percentage=10):
    nltk.download('punkt')
    
    segments = []

    for article in articles:
        # Découper le texte en phrases avec NLTK
        sentences = sent_tokenize(article)

        # Liste temporaire pour accumuler les tokens
        all_tokens = []

        # Tokeniser toutes les phrases en mots
        for sentence in sentences:
            all_tokens.extend(word_tokenize(sentence))

        # Calculer le pas pour le découpage (avec ou sans chevauchement)
        if overlap:
            step = max(1, int(max_tokens * (1 - overlap_percentage / 100)))
        else:
            step = max_tokens

        # Segmenter les tokens
        for i in range(0, len(all_tokens), step):
            segment = all_tokens[i:i + max_tokens]

            if fixed_size:
                # Padding si le segment est plus court que max_tokens
                if len(segment) < max_tokens:
                    segment += ["[PAD]"] * (max_tokens - len(segment))

            # Ajouter le segment finalisé à la liste
            segments.append(" ".join(segment))

    return segments

def segment_only(article, max_tokens=256, fixed_size=True, overlap=False, overlap_percentage=10):
    nltk.download('punkt')
    
    segments = []

    # Découper le texte en phrases avec NLTK
    sentences = sent_tokenize(article)

    # Liste temporaire pour accumuler les tokens
    all_tokens = []

    # Tokeniser toutes les phrases en mots
    for sentence in sentences:
        all_tokens.extend(word_tokenize(sentence))

    # Calculer le pas pour le découpage (avec ou sans chevauchement)
    if overlap:
        step = max(1, int(max_tokens * (1 - overlap_percentage / 100)))
    else:
        step = max_tokens

    # Segmenter les tokens
    for i in range(0, len(all_tokens), step):
        segment = all_tokens[i:i + max_tokens]

        if fixed_size:
            # Padding si le segment est plus court que max_tokens
            if len(segment) < max_tokens:
                segment += ["[PAD]"] * (max_tokens - len(segment))

        # Ajouter le segment finalisé à la liste
        segments.append(" ".join(segment))

    return segments

def segment_article_by_sentence(articles):
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    segments = []

     # Patterns à filtrer
    patterns_to_filter = [
        r'^\s*contact\s+[\d\s]+$',  # Numéros de contact
        r'^\s*tel\s*:?\s*[\d\s]+$',  # Numéros de téléphone
        r'^\s*[\d\s]{8,}$',         # Séquences de chiffres
        r'^[^a-zA-Z]*$',            # Segments sans lettres
        r'www\.',                   # URLs
        r'@',                       # Emails
        r'^\s*source\s*:',          # Citations de sources
        r'^\s*ref\s*:',             # Références
        r'^\s*contact\s*:',         # Contacts
        r'^\s*prix\s*:?\s*[\d\s€]+$',  # Prix seuls
        r'^\s*page\s+\d+\s*$',      # Numéros de page
    ]
    
    
    for article in articles:
        if not isinstance(article, str) or not article.strip():
            continue
            
        sentences = sent_tokenize(article)
        
        for sentence in sentences:
            clean_sentence = sentence.strip()
            words = word_tokenize(clean_sentence)

            # Vérifier si le segment correspond à un des patterns à filtrer
            should_skip = False
            for pattern in patterns_to_filter:
                if re.search(pattern, clean_sentence, re.IGNORECASE):
                    should_skip = True
                    break
                    
            if should_skip:
                continue

            # Ignorer les URLs et références
            if any(x in clean_sentence.lower() for x in ['www.', 'http', '.fr', '.com', '@']):
                continue

            # Ignorer les segments qui sont principalement des caractères spéciaux
            if sum(c.isalpha() for c in clean_sentence) < len(clean_sentence) * 0.3:
                continue
                
            # Ne garder que les phrases avec 3 mots ou plus
            if len(words) >= 3:
                segments.append(clean_sentence)
    
    return segments

def extraire_entites(resultats, label_recherche="magasin"):

    entites = []
    for article, data in resultats:
        for entite in data.get('entities', []):
            start, end, label = entite
            if label == label_recherche:
                nom_entite = article[start:end].strip()
                entites.append(nom_entite)
    return entites

def compter_occurrences(entites):
    return Counter(entites)

def tracer_frequence_entites(compteur, top_n=20):

    # Convertir le Counter en DataFrame
    df = pd.DataFrame(compteur.most_common(top_n), columns=['Entité', 'Occurrences'])
    
    # Configurer le style de Seaborn
    sns.set(style="whitegrid")
    
    # Créer le plot
    plt.figure(figsize=(12, 8))
    bar_plot = sns.barplot(x='Occurrences', y='Entité', data=df, palette='viridis')
    
    # Ajouter des labels
    plt.title(f"Top {top_n} Entités de Magasin de Production les Plus Fréquentes")
    plt.xlabel("Nombre d'Occurrences")
    plt.ylabel("Entité")
    
    # Afficher les valeurs au-dessus des barres
    for index, value in enumerate(df['Occurrences']):
        bar_plot.text(value, index, str(value), color='black', va="center")
    
    plt.tight_layout()
    plt.show()

def convert_annotations_to_bio(data):
    import re
    
    classes = data["classes"]
    annotations = data["annotations"]
    
    result_texts = []
    result_labels = []
    
    for text, annotation in annotations:
        # Initialise les labels avec "O" (Outside)
        tokens = re.findall(r'\S+', text)  # Tokenisation simple par espace
        labels = ["O"] * len(tokens)
        
        # Parcourt les entités et marque les labels
        for start, end, entity in annotation["entities"]:
            entity_label = f"B-{entity}"  # Marqueur Begin
            inside_label = f"I-{entity}"  # Marqueur Inside
            
            # Localise les tokens affectés par l'entité
            current_index = 0
            is_entity_started = False
            
            for i, token in enumerate(tokens):
                token_start = current_index
                token_end = current_index + len(token)
                
                # Ajuste les offsets pour correspondre à l'entité
                if start <= token_start < end or start < token_end <= end or (token_start <= start and token_end >= end):
                    if not is_entity_started:
                        labels[i] = entity_label  # Le premier token de l'entité est "B-"
                        is_entity_started = True
                    else:
                        labels[i] = inside_label  # Les autres tokens sont "I-"
                
                # Met à jour l'index courant
                current_index = token_end + 1  # +1 pour l'espace entre les tokens
        
        # Ajoute les résultats au format final
        result_texts.append(tokens)
        result_labels.append(labels)
    
    return result_texts, result_labels


from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk
import re
from typing import Union


def nettoyer_liste(df, colonne):
    df[colonne] = df[colonne].astype(str).str.strip()
    return df[colonne]

def clean_and_validate_text(segment: str, min_words: int = 6) -> Union[str, None]:

    if not isinstance(segment, str) or not segment.strip():
        return None

    # Supprimer le BOM UTF-8 et ses variantes
    segment = segment.replace('\ufeff', '')  # BOM UTF-8
    segment = segment.replace('ï»¿', '')     # BOM UTF-8 mal encodé

    special_chars = [
        '°',  # degré
        '€',  # euro
        '$',  # dollar
        '%',  # pourcentage
        '№',  # numéro
        '§',  # section
        '®',  # registered
        '™',  # trademark
        '©',  # copyright
        '±',  # plus-minus
        '#',  # hashtag
        '@',  # at
        '&',  # ampersand
        '/',  # slash
        '\\', # backslash
        '*',  # asterisk
        '+',  # plus
        '=',  # equals
        '<',  # less than
        '>',  # greater than
        '|',  # pipe
    ]
    
    # Créer un pattern pour tous les caractères spéciaux
    special_chars_pattern = '|'.join(map(re.escape, special_chars))
    
    # Ajouter des espaces avant et après les caractères spéciaux, mais éviter les espaces doubles
    segment = re.sub(f'([^ ])({special_chars_pattern})', r'\1 \2', segment)
    segment = re.sub(f'({special_chars_pattern})([^ ])', r'\1 \2', segment)

    # Supprimer uniquement les caractères de contrôle invisibles spécifiques
    segment = re.sub(r'[\u200b-\u200f\u2028-\u202f\u205f-\u206f]', '', segment)
    
    # Supprimer tous les liens internet
    segment = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', segment)
    segment = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', segment)
    
    # Supprimer toutes les balises HTML avec BeautifulSoup
    try:
        soup = BeautifulSoup(segment, 'html.parser')
        segment = soup.get_text()
    except Exception as e:
        print(f"Erreur lors du nettoyage HTML: {e}")
        return None
    
    # Nettoyer les espaces multiples et les sauts de ligne
    segment = re.sub(r'\s+', ' ', segment)
    segment = segment.strip()

    # Supprimer les BOM pouvant apparaître au milieu du texte
    segment = re.sub(r'ï»¿', '', segment)

    contact_regex = r"""
            (?:
                # Format avec labels (TÉL, FAX, etc.)
                (?:
                    (?:T[ÉE]L(?:[\s.:]+|\s*:\s*)|FAX\s*:\s*|TEL(?:[\s.:]+|\s*:\s*))
                    (?:\d{2}[.\s-]*){4,5}\d{1,2}\.?
                )
                |
                # Format avec Renseignements/Contact et toutes ses variantes
                (?:
                    (?:\(\s*)?
                    (?:Renseignements?|Rens\.?|Rens|Contactez-nous|Pour\splus\s(?:de\sdétails|d\'informations),\scontactez-nous|Ou\s+bien)
                    [\s:()\-]*
                    (?:au\s?)?
                    (?:\d{2}[.\s-]*){3,4}\d{2}\.?
                    (?:[\s()\-.]*(?:ou\s+(?:au|bien)|,|;)[\s()\-.]* (?:\d{2}[.\s-]*){3,4}\d{2}\.?)*
                    (?:\s*\))?
                )
                |
                # Format numéro simple
                (?:
                    (?:^|\s)
                    (?:\d{2}[.\s-]*){4,5}\d{1,2}\.?
                )
                |
                # Juste le mot Rens/Renseignements sans numéro
                (?:
                    (?:^|\s)
                    (?:Renseignements?|Rens\.?|Rens)
                    (?:\s|$|:|\.)
                )
            )
        """

    segment = re.sub(contact_regex, '', segment, flags=re.IGNORECASE | re.VERBOSE | re.MULTILINE)
    
    # Nettoyer les espaces multiples
    segment = re.sub(r'\s+', ' ', segment)
    
    # Nettoyer les points doublés qui pourraient rester
    segment = re.sub(r'\.+', '.', segment)
    
    # Nettoyer les espaces avant la ponctuation
    segment = re.sub(r'\s+([.,!?])', r'\1', segment)
    
    # Nettoyer les doubles espaces qui pourraient rester après le nettoyage
    segment = re.sub(r'\s+', ' ', segment)
    
    url_email_regex = r'(https?://\S+|www\.\S+|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
    
    # Regex pour corriger ". ." en "."
    double_dot_regex = r'\.\s+\.'
    
    # Supprimer les URLs et emails
    segment = re.sub(url_email_regex, '', segment)

    segment = re.sub(r"\.\s*-\s*", ". ", segment)
    segment = re.sub(r'\.(?=[a-z])', '. ', segment)
    
    # Corriger les points doubles
    segment = re.sub(double_dot_regex, '.', segment)
    # Supprimer les espaces superflus
    segment = re.sub(r'\s+', ' ', segment).strip()
    
    # Vérifier la présence de contenu non désiré
    unwanted_patterns = [
        r'<[^>]+>',  # Balises HTML résiduelles
        r'\[.*?\]',  # Contenu entre crochets
        r'\{.*?\}',  # Contenu entre accolades
        r'javascript:',
        r'@import',
        r'@media',
        r'@font-face',
        r'document\.write',
        r'window\.'
    ]
    
    for pattern in unwanted_patterns:
        if re.search(pattern, segment, re.IGNORECASE):
            return None
    
    # Vérifier la longueur minimale
    words = [w for w in segment.split() if len(w) > 1]
    if len(words) < min_words:
        return None
        
    # Vérifier que le texte contient du contenu textuel (tout type d'alphabet)
    if not re.search(r'\w', segment):
        return None
    
    return segment

# Fonction de filtrage
def filter_noisy_segments(segments):
    filtered_segments = []
    for segment in segments:
        if len(segment.split()) > 3:
            filtered_segments.append(segment)
    return filtered_segments


# Fonction de nettoyage
def clean_segments(segments, stpwords = ["cette", "ans", "aussi", "plus", "où", "tous", "comme", "depuis"]):
    # Télécharger les stopwords français
    nltk.download('stopwords')
    french_stopwords = stopwords.words('french') + stpwords
    cleaned = []
    for segment in segments:
        if segment:
            cleaned.append(" ".join([
                word for word in segment.split() if word.lower() not in french_stopwords
            ]))
    return cleaned



from transformers import AutoTokenizer
import re

def create_tokenizer(model_name="microsoft/mdeberta-v3-base"):
    return AutoTokenizer.from_pretrained(model_name)

def split_text_into_sliding_windows(text, tokenizer, max_tokens=512, stride=153):
    encoding = tokenizer(text.strip(), return_offsets_mapping=True, add_special_tokens=True)
    token_ids = encoding['input_ids']
    
    if len(token_ids) <= max_tokens:
        return [text.strip()]
    
    offset_mapping = encoding['offset_mapping']
    chunks = []
    start_token = 0
    
    while start_token < len(token_ids):
        end_token = min(start_token + max_tokens, len(token_ids))
        chunk_start = offset_mapping[start_token][0]
        chunk_end = offset_mapping[min(end_token - 1, len(offset_mapping) - 1)][1]
        chunk_text = text[chunk_start:chunk_end].strip()
        
        if end_token < len(token_ids):
            sentences = segment_sentence(chunk_text)
            if sentences and len(sentences) > 1:
                chunk_text = ' '.join(sentences[:-1])
                new_chunk_end = chunk_start + len(chunk_text)
                while end_token > start_token and offset_mapping[end_token-1][1] > new_chunk_end:
                    end_token -= 1
        
        if chunk_text:
            chunks.append(chunk_text)
        
        start_token += stride
        if end_token >= len(token_ids):
            break
    
    return chunks

def clean_segment(text):
    clean_text = re.sub(r'\s+', ' ', text)  # Supprime les espaces multiples
    clean_text = re.sub(r'\s*\.\s*-', '.-', clean_text)  # Nettoie ". -"
    return clean_text.strip()


def segment_sentence(text):
    if not text:
        return []

    EXCEPTIONS = {
        'titles': r'M\.|Mr\.|Dr\.|Mme\.',
        'abbrev': r'cf\.|etc\.|ex\.',
        'numbers': r'\d+\.\d+',
        'acronyms': r'\b(?:[a-zA-Z]\.){2,}(?=\s+[a-z])',
        'initials': r'[A-Z]\.',
        'tel': r'.\s*(?:T[ée]l[ée]phone|T[ée]l|TEL|Tel|[Rr]ens|[Rr]enseignements?)\.?:?\s*',
        'entry': r'.\s*(?:Entr[ée]e?)\.?\s*',
        'animation': r'.\s*(?:[Aa]nimations?)\.?\s*',
        'billets': r'.\s*(?:[Bb]illets?)\.?\s*',
        'ouvert': r'.\s*(?:[Ou]uvert?|[Ii]nscriptions?)\.?\s*',
        'citations': r'(?:"[^"]+")|(?:«[^»]+»)|(?:"[^"]+")'
    }
        
    protected_text = text
    exception_tokens = {}
    for name, pattern in EXCEPTIONS.items():
        matches = list(re.finditer(pattern, protected_text))
        for m in matches:
            token = f"@{name}_{len(exception_tokens)}@"
            exception_tokens[token] = m.group(0)
            protected_text = protected_text.replace(m.group(0), token)

    segments = [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[^"\s]|")', protected_text) if s.strip()]

    restored_segments = segments
    for token, original in exception_tokens.items():
        restored_segments = [clean_segment(s.replace(token, original)) for s in restored_segments]

    return restored_segments

def custom_product(entities):
    save = []
    for k in range(len(entities) - 1):
        for idx in range(k + 1, len(entities)):
            if entities[k]['text'] == entities[idx]['text'] and entities[k]['label'] == entities[idx]['label']:
                pass
            else:
                s = [entities[k], entities[idx]]
                save.append(s)
    return save

def remove_duplicates(entities):
    """Supprime les entités en double en gardant celle avec le meilleur score"""
    seen = {}
    for entity in entities:
        key = (entity["text"].lower(), entity["label"])
        if key not in seen or entity["score"] > seen[key]["score"]:
            seen[key] = entity
    return list(seen.values())


def print_segments_info(corpus):
    print("\nAnalyse des segments :")
    print("=" * 80)
    print(f"{'Article':<10} {'Nombre de segments':>20} {'Mots par segment':>50}")
    print("-" * 80)
    
    # Convertir la série en liste de segments
    segments_list = corpus['Segments'].tolist()
    
    for i, segments_str in enumerate(segments_list):
        # Enlever les caractères [ et ] et diviser en segments
        if isinstance(segments_str, str):
            segments_str = segments_str.strip('[]')
            segments = [seg.strip().strip("'") for seg in segments_str.split(',')]
        else:
            segments = segments_str
            
        # Compter le nombre de segments
        n_segments = len(segments)
        
        # Compter les mots dans chaque segment
        words_per_segment = [len(str(segment).split()) for segment in segments]
        
        # Formater la liste des nombres de mots
        words_str = ", ".join([str(x) for x in words_per_segment])
        
        print(f"{i+1:<10} {n_segments:>20} {words_str:>50}")
    
    # Statistiques globales
    all_segments = []
    for segments_str in segments_list:
        if isinstance(segments_str, str):
            segments_str = segments_str.strip('[]')
            segments = [seg.strip().strip("'") for seg in segments_str.split(',')]
            all_segments.extend(segments)
        else:
            all_segments.extend(segments_str)
    
    all_segment_lengths = [len(str(segment).split()) for segment in all_segments]
    
    print("\nStatistiques globales :")
    print("-" * 80)
    print(f"Nombre total de segments : {len(all_segments)}")
    print(f"Moyenne de mots par segment : {np.mean(all_segment_lengths):.1f}")
    print(f"Minimum de mots par segment : {min(all_segment_lengths)}")
    print(f"Maximum de mots par segment : {max(all_segment_lengths)}")

def get_segments_dataframe(corpus):
    segments_data = []
    
    for i, segments_str in enumerate(corpus['Segments']):
        if isinstance(segments_str, str):
            segments_str = segments_str.strip('[]')
            segments = [seg.strip().strip("'") for seg in segments_str.split(',')]
        else:
            segments = segments_str
            
        for j, segment in enumerate(segments):
            segments_data.append({
                'Article': i+1,
                'Segment': j+1,
                'Nombre_de_mots': len(str(segment).split()),
                'Texte': str(segment)[:100] + '...' if len(str(segment)) > 100 else str(segment)
            })
    
    df = pd.DataFrame(segments_data)
    return df

def segment_recursively(segment, segmenter, max_words=256):
    previous_length = len(segment.split())
    
    while previous_length > max_words:
        sub_segments, _, _ = segmenter.segment_text(segment)
        new_segments = [' '.join(s) for s in sub_segments]
        new_total_length = sum(len(s.split()) for s in new_segments)

        if all(len(s.split()) <= max_words for s in new_segments) or new_total_length >= previous_length:
            return new_segments  

        previous_length = new_total_length
        segment = ' '.join(new_segments)

    return [segment]

# def split_thematic_segment(corpus, segmenter):
    
#     All_segments = []
#     for i, article in corpus["Article"].items():
#         generated_segments, coherence_scores, boundaries = segmenter.segment_text(article)
#         merged_segments_list = []
#         for idx, segment in enumerate(generated_segments):
#             merged_segments = ' '.join(segment) 
#             refined_segments = segment_recursively(merged_segments, segmenter, max_words=384)
            
#             for refined_segment in refined_segments:
#                 merged_segments_list.append(refined_segment)
                
#         All_segments.append(merged_segments_list)   
#     return  All_segments

from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

def split_thematic_segment(
    corpus: List[Tuple[str, dict]],
    segmenter
) -> List[List[str]]:

    all_segments = []

    for idx, (article_text, _) in enumerate(corpus):
        try:
            generated_segments, _, _ = segmenter.segment_text(article_text)
            merged_segments_list = []

            for segment in generated_segments:
                merged_text = ' '.join(segment)
                refined_segments = segment_recursively(merged_text, segmenter, max_words=384)
                merged_segments_list.extend(refined_segments)

            all_segments.append(merged_segments_list)

        except Exception as e:
            logger.exception(f"Error while processing article index {idx}: {e}")
            all_segments.append([])  # append empty list to preserve structure

    return all_segments

def split_thematic_segment_labelstudio(
    label_studio_data: List[dict],
    segmenter
) -> List[dict]:
    """
    Modifie les données Label Studio pour segmenter thématiquement le texte
    et recalcule les spans.
    
    Args:
        label_studio_data: Liste des données au format Label Studio
        segmenter: Instance du segmenteur thématique
    
    Returns:
        Liste des données Label Studio avec segments et spans recalculés
    """
    
    def recalculate_spans(segment_text: str, original_entities: List[dict], segment_start_pos: int) -> List[dict]:
        """Recalcule les spans des entités pour un segment donné"""
        segment_entities = []
        segment_end_pos = segment_start_pos + len(segment_text)
        
        for entity in original_entities:
            entity_start = entity['start']
            entity_end = entity['end']
            
            # Vérifier si l'entité est dans ce segment
            if entity_start < segment_end_pos and entity_end > segment_start_pos:
                # Ajuster les positions relatives au segment
                adjusted_start = max(0, entity_start - segment_start_pos)
                adjusted_end = min(len(segment_text), entity_end - segment_start_pos)
                
                # Vérifier que l'entité ajustée est valide
                if adjusted_start < adjusted_end:
                    adjusted_entity = entity.copy()
                    adjusted_entity['start'] = adjusted_start
                    adjusted_entity['end'] = adjusted_end
                    adjusted_entity['text'] = segment_text[adjusted_start:adjusted_end]
                    segment_entities.append(adjusted_entity)
        
        return segment_entities
    
    result_data = []
    
    for idx, item in enumerate(label_studio_data):
        try:
            # Extraire le texte et les annotations
            article_text = item['data']['text']
            original_entities = []
            
            # Extraire les entités des annotations Label Studio
            if 'annotations' in item and item['annotations']:
                for annotation in item['annotations']:
                    if 'result' in annotation:
                        for result in annotation['result']:
                            if result.get('type') == 'labels':
                                original_entities.append({
                                    'start': result['value']['start'],
                                    'end': result['value']['end'],
                                    'text': result['value']['text'],
                                    'labels': result['value']['labels']
                                })
            
            # Segmentation thématique
            generated_segments, _, _ = segmenter.segment_text(article_text)
            
            # Traitement de chaque segment
            for segment_idx, segment in enumerate(generated_segments):
                merged_text = ' '.join(segment)
                
                # Trouver la position du segment dans le texte original
                segment_start_pos = article_text.find(merged_text)
                if segment_start_pos == -1:
                    # Si le segment exact n'est pas trouvé, essayer de le localiser approximativement
                    segment_start_pos = 0
                    for i, prev_segment in enumerate(generated_segments[:segment_idx]):
                        prev_merged = ' '.join(prev_segment)
                        segment_start_pos += len(prev_merged) + 1  # +1 pour l'espace
                
                # Appliquer la segmentation récursive si nécessaire
                refined_segments = segment_recursively(merged_text, segmenter)
                
                # Traiter chaque segment raffiné
                current_pos = segment_start_pos
                for refined_segment in refined_segments:
                    # Recalculer les spans pour ce segment
                    segment_entities = recalculate_spans(refined_segment, original_entities, current_pos)
                    
                    # Créer un nouvel item Label Studio pour ce segment
                    new_item = {
                        'data': {'text': refined_segment},
                        'annotations': [{
                            'result': []
                        }]
                    }
                    
                    # Ajouter les entités au format Label Studio
                    for entity in segment_entities:
                        new_item['annotations'][0]['result'].append({
                            'type': 'labels',
                            'value': {
                                'start': entity['start'],
                                'end': entity['end'],
                                'text': entity['text'],
                                'labels': entity['labels']
                            }
                        })
                    
                    result_data.append(new_item)
                    current_pos += len(refined_segment) + 1  # +1 pour l'espace
        
        except Exception as e:
            logger.exception(f"Error while processing article index {idx}: {e}")
            # Ajouter un item vide pour préserver la structure
            result_data.append({
                'data': {'text': ''},
                'annotations': [{'result': []}]
            })
    
    return result_data


import re
import pandas as pd
import ast

def nettoyer_dates_surface(corpus):
    # Fonction pour vérifier si un texte à une position donnée correspond à une surface
    def est_surface(texte, debut, fin):
        # Extraire le texte à la position donnée
        extrait = texte[debut:fin]
        
        # Vérifier si l'extrait contient uniquement des chiffres
        if extrait.isdigit():
            # Vérifier si le texte après l'extrait contient "m2", "m²", etc.
            apres = texte[fin:fin+5]  # Regarder quelques caractères après
            pattern_surface = r'\s*(?:m2|m²|mètres?[ -]?carrés?)'
            return bool(re.match(pattern_surface, apres))
        
        return False
    
    # Copier le corpus pour ne pas modifier l'original
    corpus_nettoye = corpus.copy()
    
    # Parcourir chaque ligne du corpus
    for index, row in corpus_nettoye.iterrows():
        if pd.isna(row['heideltime_dates_with_spans']) or row['heideltime_dates_with_spans'] == '[]':
            continue
        
        # Convertir la chaîne en liste de dictionnaires
        try:
            dates_spans = ast.literal_eval(row['heideltime_dates_with_spans'])
        except:
            # Si la conversion échoue, passer à la ligne suivante
            continue
        
        article_text = row['Article']
        
        # Filtrer les dates dont les spans ne correspondent pas à des surfaces
        dates_spans_nettoyes = []
        for date_span in dates_spans:
            debut = date_span['span'][0]
            fin = date_span['span'][1]
            
            if not est_surface(article_text, debut, fin):
                dates_spans_nettoyes.append(date_span)
        
        # Mettre à jour la colonne avec les dates nettoyées
        corpus_nettoye.at[index, 'heideltime_dates_with_spans'] = str(dates_spans_nettoyes)
    
    return corpus_nettoye


import pandas as pd
import ast
import json

def nettoyer_chevauchements_mag_localisation_dates(corpus):
    # Liste des termes à supprimer de la colonne Localisation
    termes_a_supprimer = ["circuit-court","Bonjour Bugey","État","Boutique paysans","Conseil Général", 
                          "Régional", "Chambre d'agriculture", "Bourg Habitat", "Région", "M Thiévon",
                          "Arche Fermière","Serton","communauté de communes de Belle-Ile",
                          "NOTE Ferme du Rocher blanc","GAEC Thévenon","Poulets de garrigue","Rhône-Alpes - Agriculteurs",
                          "réseau Civam","Civam","Centre","Tél","Rens","Tel","Bio","AMAP","è Pratique","Européennes"]
    
    # Copier le corpus pour ne pas modifier l'original
    corpus_nettoye = corpus.copy()
    
    # Parcourir chaque ligne du corpus
    for index, row in corpus_nettoye.iterrows():
        # Récupérer les spans de MAG s'ils existent
        spans_mag = []
        if 'MAG' in row and row['MAG'] is not None and not (isinstance(row['MAG'], float) and pd.isna(row['MAG'])):
            try:
                mags = ast.literal_eval(row['MAG']) if isinstance(row['MAG'], str) else row['MAG']
                for mag in mags:
                    if 'span' in mag:
                        debut, fin = mag['span'][0], mag['span'][1]
                        spans_mag.append((debut, fin))
            except:
                # Si erreur, continuer avec une liste vide
                spans_mag = []
        
        # Traitement des dates si elles existent
        if 'heideltime_dates_with_spans' in row and row['heideltime_dates_with_spans'] is not None and not (isinstance(row['heideltime_dates_with_spans'], float) and pd.isna(row['heideltime_dates_with_spans'])):
            try:
                dates = ast.literal_eval(row['heideltime_dates_with_spans']) if isinstance(row['heideltime_dates_with_spans'], str) else row['heideltime_dates_with_spans']
            except:
                dates = []
                
            # Filtrer les dates qui ne chevauchent pas les spans de MAG
            if spans_mag:
                dates_filtrees = []
                for date in dates:
                    if 'span' in date:
                        debut_date, fin_date = date['span'][0], date['span'][1]
                        
                        # Vérifier s'il y a chevauchement avec un span de MAG
                        chevauchement = False
                        for debut_mag, fin_mag in spans_mag:
                            if (debut_date < fin_mag and fin_date > debut_mag):
                                chevauchement = True
                                break
                        
                        # Si pas de chevauchement, garder la date
                        if not chevauchement:
                            dates_filtrees.append(date)
                    else:
                        dates_filtrees.append(date)
                        
                # Mettre à jour la colonne des dates
                corpus_nettoye.at[index, 'heideltime_dates_with_spans'] = json.dumps(dates_filtrees) if dates_filtrees else "[]"
        
        # Traitement des localisations si elles existent
        if 'Localisation' in row and row['Localisation'] is not None and not (isinstance(row['Localisation'], float) and pd.isna(row['Localisation'])):
            try:
                localisations = ast.literal_eval(row['Localisation']) if isinstance(row['Localisation'], str) else row['Localisation']
            except:
                continue
                
            # Filtrer les localisations trop courtes et les termes à supprimer
            localisations = [
                loc for loc in localisations 
                if 'text' in loc and len(loc['text'].strip()) > 1 and not any(terme.lower() in loc['text'].lower() for terme in termes_a_supprimer)
            ]
            
            # Si pas de spans MAG, mettre à jour et continuer
            if not spans_mag:
                corpus_nettoye.at[index, 'Localisation'] = json.dumps(localisations) if localisations else "[]"
                continue
            
            # Filtrer les localisations qui ne chevauchent pas les spans de MAG
            localisations_filtrees = []
            for loc in localisations:
                if 'span' in loc:
                    debut_loc, fin_loc = loc['span'][0], loc['span'][1]
                    
                    # Vérifier s'il y a chevauchement avec un span de MAG
                    chevauchement = False
                    for debut_mag, fin_mag in spans_mag:
                        if (debut_loc < fin_mag and fin_loc > debut_mag):
                            chevauchement = True
                            break
                    
                    # Si pas de chevauchement, garder la localisation
                    if not chevauchement:
                        localisations_filtrees.append(loc)
                else:
                    localisations_filtrees.append(loc)
            
            # Mettre à jour la colonne Localisation
            corpus_nettoye.at[index, 'Localisation'] = json.dumps(localisations_filtrees) if localisations_filtrees else "[]"
    
    return corpus_nettoye
