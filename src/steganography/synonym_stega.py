import random

from typing import Any, Dict, List, Tuple



import numpy as np

from nltk import pos_tag

from nltk.corpus import wordnet

from nltk.stem import WordNetLemmatizer

from sentence_transformers import SentenceTransformer



from src.text_utils import detokenize_words, tokenize_words





class SynonymSteganography:

    """Word-level synonym substitution with a local-context similarity filter.



    The semantic filter reduces obviously unsuitable candidates but does not

    guarantee grammaticality or exact meaning preservation. Validate samples.

    """



    def __init__(

        self,

        max_substitutions: int = 8,

        substitution_probability: float = 0.25,

        seed: int = 42,

        similarity_threshold: float = 0.88,

        context_window: int = 5,

        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",

    ):

        self.max_substitutions = max_substitutions

        self.substitution_probability = substitution_probability

        self.random = random.Random(seed)

        self.lemmatizer = WordNetLemmatizer()

        self.similarity_threshold = similarity_threshold

        self.context_window = context_window

        # Load once per encoder instance, not once per word.

        self.semantic_model = SentenceTransformer(embedding_model_name)

        self.stop_words = {

            "the", "a", "an", "and", "or", "but", "is", "are", "was",

            "were", "be", "been", "to", "of", "in", "on", "for", "with",

            "as", "at", "by", "from", "this", "that", "it", "its", "i",

            "we", "you", "he", "she", "they", "them", "my", "your",

            "our", "their",

        }



    def _is_candidate_word(self, word: str) -> bool:

        return word.isalpha() and len(word) >= 4 and word.lower() not in self.stop_words



    @staticmethod

    def _wordnet_pos(treebank_tag: str):

        if treebank_tag.startswith("N"):

            return wordnet.NOUN

        if treebank_tag.startswith("V"):

            return wordnet.VERB

        if treebank_tag.startswith("J"):

            return wordnet.ADJ

        if treebank_tag.startswith("R"):

            return wordnet.ADV

        return None



    def _inflect_like_original(self, original: str, candidate: str, pos: str) -> str:

        """Conservative suffix handling; not a full morphology system."""

        word = original.lower()

        candidate = candidate.lower()



        if pos == wordnet.VERB:

            if word.endswith("ing"):

                if candidate.endswith("e"):

                    candidate = candidate[:-1]

                return candidate + "ing"

            if word.endswith("ied"):

                return candidate[:-1] + "ied" if candidate.endswith("y") else candidate + "ied"

            if word.endswith("ed"):

                return candidate + "d" if candidate.endswith("e") else candidate + "ed"

            if word.endswith("ies"):

                return candidate[:-1] + "ies" if candidate.endswith("y") else candidate + "ies"

            if word.endswith("s") and not word.endswith("ss"):

                return candidate[:-1] + "ies" if candidate.endswith("y") else candidate + "s"



        if pos == wordnet.NOUN and word.endswith("s") and not word.endswith("ss"):

            if candidate.endswith("y"):

                return candidate[:-1] + "ies"

            if candidate.endswith(("s", "x", "z", "ch", "sh")):

                return candidate + "es"

            return candidate + "s"



        if pos == wordnet.ADJ:

            if word.endswith("iest") and candidate.endswith("y"):

                return candidate[:-1] + "iest"

            if word.endswith("est"):

                return candidate + "est"

            if word.endswith("ier") and candidate.endswith("y"):

                return candidate[:-1] + "ier"

            if word.endswith("er"):

                return candidate + "er"



        return candidate



    @staticmethod

    def _match_case(original: str, replacement: str) -> str:

        if original.isupper():

            return replacement.upper()

        if original.istitle():

            return replacement.capitalize()

        return replacement.lower()



    def _get_synonyms(self, word: str, treebank_tag: str) -> List[str]:

        pos = self._wordnet_pos(treebank_tag)

        if pos is None:

            return []



        word_lower = word.lower()

        lemma = self.lemmatizer.lemmatize(word_lower, pos=pos)

        candidates = set()



        for synset in wordnet.synsets(lemma, pos=pos):

            if synset is None:

                continue

            get_lemmas = getattr(synset, "lemmas", None)

            if get_lemmas is None:

                continue

            for lemma_obj in get_lemmas():

                get_name = getattr(lemma_obj, "name", None)

                if get_name is None:

                    continue

                synonym = str(get_name()).replace("_", " ").lower()

                if (

                    " " in synonym

                    or not synonym.isalpha()

                    or len(synonym) < 3

                    or synonym == lemma

                    or synonym == word_lower

                ):

                    continue

                inflected = self._inflect_like_original(word_lower, synonym, pos)

                if inflected.isalpha() and inflected != word_lower:

                    candidates.add(inflected)

        return sorted(candidates)



    def _context(self, tokens: List[str], index: int) -> str:

        start = max(0, index - self.context_window)

        end = min(len(tokens), index + self.context_window + 1)

        return detokenize_words(tokens[start:end])





    def _rank_contextual_candidates_batch(

        self,

        tokens: List[str],

        candidate_groups: List[Tuple[int, List[str]]],

    ) -> Dict[int, List[Tuple[str, float]]]:

        """Rank candidates for multiple words in one model call."""



        if not candidate_groups:

            return {}



        contexts = []

        spans = {}



        for index, candidates in candidate_groups:

            start = len(contexts)



            # Baseline context for this word.

            contexts.append(self._context(tokens, index))



            # Context after replacing this word with each candidate.

            for candidate in candidates:

                trial_tokens = tokens.copy()

                trial_tokens[index] = candidate

                contexts.append(self._context(trial_tokens, index))



            spans[index] = (start, candidates)



        embeddings = self.semantic_model.encode(

            contexts,

            convert_to_numpy=True,

            normalize_embeddings=True,

            show_progress_bar=False,

            batch_size=32,

        )



        results = {}



        for index, (start, candidates) in spans.items():

            original_embedding = embeddings[start]



            candidate_embeddings = embeddings[

                start + 1 : start + 1 + len(candidates)

            ]



            scores = np.matmul(

                candidate_embeddings,

                original_embedding,

            )



            ranked = sorted(

                zip(candidates, scores.tolist()),

                key=lambda item: item[1],

                reverse=True,

            )



            results[index] = [

                (word, float(score))

                for word, score in ranked

            ]



        return results





    def encode(self, text: str) -> Tuple[str, Dict[str, Any]]:

        """Encode text using POS-aware, context-preserving substitutions."""



        tokens = tokenize_words(text)

        tagged_words = pos_tag(tokens)



        candidate_indices = [

            i

            for i, token in enumerate(tokens)

            if self._is_candidate_word(token)

        ]



        self.random.shuffle(candidate_indices)



        # Gather candidate groups before running the semantic model.

        candidate_groups = []

        max_candidates_per_word = 12



        for index in candidate_indices:

            if len(candidate_groups) >= self.max_substitutions:

                break



            if self.random.random() > self.substitution_probability:

                continue



            original_word = tokens[index]

            tag = tagged_words[index][1]



            synonyms = self._get_synonyms(original_word, tag)



            if not synonyms:

                continue



            # Limit computation while retaining candidate diversity.

            if len(synonyms) > max_candidates_per_word:

                synonyms = self.random.sample(

                    synonyms,

                    max_candidates_per_word,

                )



            candidate_groups.append((index, synonyms))



        # One sentence-transformer call per review, rather than per word.

        ranked_by_index = self._rank_contextual_candidates_batch(

            tokens,

            candidate_groups,

        )



        modified_positions = []

        substitutions = []



        for index, _ in candidate_groups:

            if len(modified_positions) >= self.max_substitutions:

                break



            original_word = tokens[index]

            ranked = ranked_by_index.get(index, [])



            acceptable = [

                (word, score)

                for word, score in ranked

                if score >= self.similarity_threshold

            ]



            if not acceptable:

                continue



            # Random choice among the three highest-scoring candidates.

            top_candidates = acceptable[:3]



            replacement_lower, similarity = self.random.choice(

                top_candidates

            )



            replacement = self._match_case(

                original_word,

                replacement_lower,

            )



            tokens[index] = replacement

            modified_positions.append(index)



            substitutions.append({

                "token_index": index,

                "original_word": original_word,

                "replacement_word": replacement,

                "pos_tag": tagged_words[index][1],

                "context_similarity": round(similarity, 4),

            })



        stego_text = detokenize_words(tokens)



        metadata: Dict[str, Any] = {

            "algorithm": "synonym",

            "modified_positions": modified_positions,

            "substitutions": substitutions,

            "num_substitutions": len(modified_positions),

            "original_text": text,

            "stego_text": stego_text,

            "similarity_threshold": self.similarity_threshold,

        }



        return stego_text, metadata
