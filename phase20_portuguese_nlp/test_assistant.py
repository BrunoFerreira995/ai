import unittest

from phase20_portuguese_nlp.assistant import (
    PortugueseAssistant,
    evaluate_classifier,
    extract_entities,
    load_intent_dataset,
    normalize_text,
    tokenize,
)
from phase20_portuguese_nlp.lexicon import canonicalize_text
from phase20_portuguese_nlp.education import EDUCATIONAL_KNOWLEDGE, detect_subject
from phase20_portuguese_nlp.portuguese_language import (
    correct_common_orthography,
    interpret_text,
    morphological_analysis,
    portuguese_topic_summary,
    semantic_relations,
    syntactic_analysis,
)


class PortugueseNLPTest(unittest.TestCase):
    def setUp(self):
        texts, intents = load_intent_dataset("phase20_portuguese_nlp/data/intents.jsonl")
        self.assistant = PortugueseAssistant().fit(texts, intents)

    def test_normalization_and_tokenization(self):
        self.assertEqual(normalize_text("Olá,  MUNDO!"), "olá mundo")
        self.assertEqual(tokenize("Ação rápida"), ["ação", "rápida"])
        self.assertIn("executar", canonicalize_text("rodar e executar o modelo"))
        self.assertEqual(normalize_text("Quero estudar matemática"), "querer estudar matemática")

    def test_intent_entities_context_and_response(self):
        result = self.assistant.respond("Preciso estudar matemática")
        self.assertEqual(result["intent"], "educacao_matemática")
        self.assertIn("Matemática", result["response"])
        entities = extract_entities("Envie para aluno@example.com em 10/08/2026")
        self.assertEqual({entity["type"] for entity in entities}, {"EMAIL", "DATE"})
        self.assertEqual(len(self.assistant.history), 1)

    def test_evaluation_metrics(self):
        true = ["saudacao", "despedida", "treinamento"]
        predicted = ["saudacao", "despedida", "saudacao"]
        metrics = evaluate_classifier(true, predicted)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertIn("f1", metrics)

    def test_educational_knowledge(self):
        self.assertEqual(detect_subject("quero estudar física"), "Física")
        self.assertEqual(len(EDUCATIONAL_KNOWLEDGE), 15)
        result = self.assistant.respond("preciso de ajuda com redação")
        self.assertEqual(result["subject"], "Redação")
        self.assertIn("argumentação", result["response"])

    def test_portuguese_language_topics(self):
        self.assertIn("tema", portuguese_topic_summary("Leitura e interpretação de textos"))
        self.assertEqual(correct_common_orthography("Voce estuda matematica"), "você estuda matemática")
        self.assertEqual(interpret_text("A escola ensina. O aluno aprende.")["word_count"], 6)
        self.assertTrue(morphological_analysis("estudar")[-1]["category"].startswith("verbo"))
        self.assertIn("verb_hints", syntactic_analysis("alunos estudam"))
        self.assertIn("synonyms", semantic_relations("feliz"))


if __name__ == "__main__":
    unittest.main()
