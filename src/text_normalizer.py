"""
Text Normalizer Module
----------------------

This module is designed to clean and normalize STT transcription results.
Features:
- Case normalization
- Removal of unwanted characters
- Dictionary-based replacements for cultural terms
- Fuzzy matching for common typos
- Capitalization enforcement for important local terms
- Extended dictionary for Reog Ponorogo cultural context
"""

import re
import difflib
from typing import Dict, List


class TextNormalizer:
    def __init__(self):
        # Dictionary of known misheard or misspelled terms -> correct form
        self.replacements: Dict[str, str] = {
            # Core Reog Ponorogo terms
            "riyadh ponderogo": "Reog Ponorogo",
            "reog ponderogo": "Reog Ponorogo",
            "reok ponorogo": "Reog Ponorogo",
            "reog ponorogo": "Reog Ponorogo",
            "ponorogo reg": "Reog Ponorogo",
            "reog": "Reog",
            "ponorogo": "Ponorogo",

            # Characters and figures
            "klono sewandono": "Raja Klono Sewandono",
            "bantarangin": "Kerajaan Bantarangin",
            "kediri": "Kerajaan Kediri",
            "ragil kuning": "Dewi Ragil Kuning",
            "putri sanggalangit": "Putri Sanggalangit",
            "singabarong": "Raja Singabarong",
            "bujanganom": "Bujanganom",
            "warok": "Warok",

            # Props and cultural items
            "dadak merak": "Dadak Merak",
            "ganongan": "Ganongan",
            "bujang ganong": "Bujang Ganong",
            "jaran kepang": "Jaran Kepang",
            "jathilan": "Jathilan",
            "pecut": "Pecut",
            "cemeti": "Cemeti",
            "barongan": "Barongan",

            # Musical instruments
            "gamelan": "Gamelan",
            "angklung reog": "Angklung Reog",
            "terompet reog": "Terompet Reog",
            "kongkil": "Kongkil",
            "kendang": "Kendang",
            "saron": "Saron",
            "gong": "Gong",
            "kempul": "Kempul",

            # Institutions and recognition
            "unesco": "UNESCO",
            "creative cities network": "Creative Cities Network",
            "museum reog ponorogo": "Museum Reog Ponorogo",
        }

        # Important cultural terms to preserve capitalization
        self.important_terms: List[str] = [
            "Reog Ponorogo",
            "Ponorogo",
            "Raja Klono Sewandono",
            "Kerajaan Bantarangin",
            "Kerajaan Kediri",
            "Dewi Ragil Kuning",
            "Putri Sanggalangit",
            "Raja Singabarong",
            "Bujanganom",
            "Warok",
            "Dadak Merak",
            "Ganongan",
            "Bujang Ganong",
            "Jaran Kepang",
            "Jathilan",
            "Pecut",
            "Cemeti",
            "Barongan",
            "Gamelan",
            "Angklung Reog",
            "Terompet Reog",
            "Kongkil",
            "Kendang",
            "Saron",
            "Gong",
            "Kempul",
            "UNESCO",
            "Creative Cities Network",
            "Museum Reog Ponorogo",
        ]

    def normalize_case(self, text: str) -> str:
        """Convert text to lowercase for easier matching"""
        return text.lower()

    def remove_unwanted_chars(self, text: str) -> str:
        """Remove non-alphanumeric characters except basic punctuation"""
        return re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)

    def apply_dictionary(self, text: str) -> str:
        """Replace known misheard/misspelled terms with correct forms"""
        normalized = text
        for wrong, correct in self.replacements.items():
            normalized = normalized.replace(wrong, correct)
        return normalized

    def fuzzy_replace(self, text: str, cutoff: float = 0.8) -> str:
        """
        Use fuzzy matching to replace words similar to important terms.
        cutoff = similarity threshold (0.0 - 1.0)
        """
        words = text.split()
        corrected_words = []

        for word in words:
            matches = difflib.get_close_matches(word, self.important_terms, n=1, cutoff=cutoff)
            if matches:
                corrected_words.append(matches[0])
            else:
                corrected_words.append(word)

        return " ".join(corrected_words)

    def capitalize_terms(self, text: str) -> str:
        """Ensure important cultural terms are capitalized correctly"""
        normalized = text
        for term in self.important_terms:
            normalized = re.sub(term.lower(), term, normalized, flags=re.IGNORECASE)
        return normalized

    def normalize(self, text: str) -> str:
        """
        Full normalization pipeline:
        1. Lowercase
        2. Remove unwanted characters
        3. Apply dictionary replacements
        4. Fuzzy replace
        5. Capitalize important terms
        """
        step1 = self.normalize_case(text)
        step2 = self.remove_unwanted_chars(step1)
        step3 = self.apply_dictionary(step2)
        step4 = self.fuzzy_replace(step3)
        step5 = self.capitalize_terms(step4)
        return step5.strip()


# Example usage
if __name__ == "__main__":
    normalizer = TextNormalizer()

    raw_texts = [
        "Hari ini kita menonton Riyadh Ponderogo di alun-alun.",
        "Pertunjukan Reok Ponorogo sangat meriah!",
        "Singabarong menghadang Raja Klono Sewandono.",
        "Dadak Merak terlihat megah dengan bulu merak.",
        "Festival Reog Ponorogo diakui oleh unesco.",
    ]

    for raw in raw_texts:
        print("\nRaw:", raw)
        print("Normalized:", normalizer.normalize(raw))