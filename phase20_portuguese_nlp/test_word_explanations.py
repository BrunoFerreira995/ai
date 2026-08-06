import tempfile
import unittest
from pathlib import Path

from .word_explanations import contiguous_subsequences, explain_word, write_word_explanations


class WordExplanationTests(unittest.TestCase):
    def test_subsequences_are_contiguous_and_unique(self):
        self.assertEqual(contiguous_subsequences("casa", 2, 3), ["ca", "as", "sa", "cas", "asa"])

    def test_explanation_contains_word_parts(self):
        result = explain_word("matemática")
        self.assertIn("matemática", result["answer"])
        self.assertIn("ma", result["subsequences"])

    def test_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "dictionary"
            source.mkdir()
            (source / "lexico").write_text("casa\nmatemática\n", encoding="utf-8")
            output = Path(directory) / "words.jsonl"
            self.assertEqual(write_word_explanations(output, source), 2)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 2)


if __name__ == "__main__":
    unittest.main()

