
import random
from typing import Any, Dict, List, Tuple

from nltk import pos_tag
from nltk.corpus import wordnet

from src.text_utils import detokenize_words, tokenize_words


class SynonymSteganography:
    """
    Word-level synonym substitution steganography.

    Selected words are replaced with single-word synonyms obtained
    from WordNet. The encoder records modifications for explainability
    and localization experiments.
    """

    def __init__(
        self,
        max_substitutions: int = 8,
        substitution_probability: float = 0.25,
        seed: int = 42,
    ):
        self.max_substitutions = max_substitutions
        self.substitution_probability = substitution_probability
        self.random = random.Random(seed)

        self.stop_words = {
            "the", "a", "an", "and", "or", "but",
            "is", "are", "was", "were", "be", "been",
            "to", "of", "in", "on", "for", "with",
            "as", "at", "by", "from", "this", "that",
            "it", "its", "i", "we", "you", "he",
            "she", "they", "them", "my", "your",
            "our", "their",
        }

    def _is_candidate_word(self, word: str) -> bool:
        """Check whether a token can be considered for replacement."""

        if not word.isalpha():
            return False

        if len(word) < 4:
            return False

        if word.lower() in self.stop_words:
            return False

        return True

    def _get_synonyms(self, word: str) -> List[str]:
        """
        Retrieve single-word synonyms with a compatible part of speech.

        Note: POS matching and suffix filtering are basic safeguards;
        they do not guarantee contextual or grammatical correctness.
        """

        word_lower = word.lower()

        tagged = pos_tag([word_lower])
        word_tag = tagged[0][1]

        if word_tag.startswith("N"):
            target_pos = wordnet.NOUN
        elif word_tag.startswith("V"):
            target_pos = wordnet.VERB
        elif word_tag.startswith("J"):
            target_pos = wordnet.ADJ
        elif word_tag.startswith("R"):
            target_pos = wordnet.ADV
        else:
            return []

        candidates = set()

        for synset in wordnet.synsets(
            word_lower,
            pos=target_pos,
        ):
            if synset is None:
                continue

            get_lemmas = getattr(synset, "lemmas", None)
            if get_lemmas is None:
                continue

            for lemma in get_lemmas():
                get_name = getattr(lemma, "name", None)
                if get_name is None:
                    continue

                synonym = str(get_name()).replace("_", " ").lower()

                # Only allow single-word alphabetic replacements.
                if " " in synonym or not synonym.isalpha():
                    continue

                if synonym == word_lower:
                    continue

                # Basic protection against incompatible verb forms.
                if target_pos == wordnet.VERB:
                    if (
                        word_lower.endswith("ing")
                        and not synonym.endswith("ing")
                    ):
                        continue

                    if (
                        word_lower.endswith("ed")
                        and not synonym.endswith("ed")
                    ):
                        continue

                    if (
                        word_lower.endswith("s")
                        and not word_lower.endswith("ss")
                        and not synonym.endswith("s")
                    ):
                        continue

                candidates.add(synonym)

        return sorted(candidates)

    def _match_case(
        self,
        original: str,
        replacement: str,
    ) -> str:
        """Preserve the capitalization style of the original word."""

        if original.isupper():
            return replacement.upper()

        if original.istitle():
            return replacement.capitalize()

        return replacement.lower()

    def encode(
        self,
        text: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Encode text using synonym substitutions.

        Returns:
            A tuple containing the modified text and substitution metadata.
        """

        tokens = tokenize_words(text)

        candidate_indices = [
            index
            for index, token in enumerate(tokens)
            if self._is_candidate_word(token)
        ]

        self.random.shuffle(candidate_indices)

        modified_positions = []
        substitutions = []

        for index in candidate_indices:
            if len(modified_positions) >= self.max_substitutions:
                break

            if self.random.random() > self.substitution_probability:
                continue

            original_word = tokens[index]
            synonyms = self._get_synonyms(original_word)

            if not synonyms:
                continue

            replacement = self.random.choice(synonyms)

            replacement = self._match_case(
                original=original_word,
                replacement=replacement,
            )

            tokens[index] = replacement
            modified_positions.append(index)

            substitutions.append(
                {
                    "token_index": index,
                    "original_word": original_word,
                    "replacement_word": replacement,
                }
            )

        stego_text = detokenize_words(tokens)

        metadata: Dict[str, Any] = {
            "algorithm": "synonym",
            "modified_positions": modified_positions,
            "substitutions": substitutions,
            "num_substitutions": len(modified_positions),
            "original_text": text,
            "stego_text": stego_text,
        }

        return stego_text, metadata
