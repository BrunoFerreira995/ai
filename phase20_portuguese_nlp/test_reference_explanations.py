import json
import tempfile
import unittest
from pathlib import Path

from .reference_explanations import explain_reference, write_reference_explanations


class ReferenceExplanationTests(unittest.TestCase):
    def test_explanation_is_metadata_only(self):
        row = explain_reference(
            "Matemática — álgebra",
            {"title": "Um artigo", "authors": "Uma Autora", "year": "2024", "doi": "10.1234/test"},
            1,
        )
        self.assertIn("álgebra", row["answer"])
        self.assertIn("10.1234/test", row["answer"])
        self.assertIn("necessário ler o artigo", row["answer"])

    def test_writes_all_references(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "references.json"
            source.write_text(json.dumps({"Matemática — álgebra": [{"title": "A", "doi": "1"}, {"title": "B", "doi": "2"}]}), encoding="utf-8")
            output = Path(directory) / "explanations.jsonl"
            self.assertEqual(write_reference_explanations(source, output), 2)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
