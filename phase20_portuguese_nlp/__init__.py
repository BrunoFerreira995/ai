"""Portuguese-language understanding foundation."""

from .assistant import PortugueseAssistant, evaluate_classifier, normalize_text, tokenize
from .lexicon import SYNONYM_DICTIONARY, VERB_DICTIONARY, canonicalize_text
from .education import EDUCATIONAL_KNOWLEDGE, detect_subject, subject_response
from .portuguese_language import (
    correct_common_orthography,
    interpret_text,
    morphological_analysis,
    portuguese_topic_summary,
    semantic_relations,
    syntactic_analysis,
)
from .literacy import analyze_word, literacy_response, split_syllables
from .dictionary import is_portuguese_word, load_portuguese_words

__all__ = [
    "PortugueseAssistant",
    "SYNONYM_DICTIONARY",
    "VERB_DICTIONARY",
    "canonicalize_text",
    "EDUCATIONAL_KNOWLEDGE",
    "detect_subject",
    "evaluate_classifier",
    "normalize_text",
    "tokenize",
    "subject_response",
    "correct_common_orthography",
    "interpret_text",
    "morphological_analysis",
    "portuguese_topic_summary",
    "semantic_relations",
    "syntactic_analysis",
    "analyze_word",
    "literacy_response",
    "split_syllables",
    "is_portuguese_word",
    "load_portuguese_words",
]
