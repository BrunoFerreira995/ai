import unittest
from pathlib import Path

from .build_qa_dataset import build_reference_rows


class ReferenceDatasetTests(unittest.TestCase):
    def test_every_subtopic_has_three_reference_questions(self):
        rows = build_reference_rows()
        self.assertEqual(len(rows), 75 * 3)
        self.assertTrue(any("Matemática" in row["question"] for row in rows))
        self.assertTrue(any("orientação profissional" in row["question"].lower() for row in rows))
        self.assertTrue(all(row["answer"].count("https://") == 4 for row in rows))

    def test_collected_dataset_has_sixty_per_subtopic(self):
        path = Path("data/educational_references_60.json")
        if not path.is_file():
            self.skipTest("coleção Crossref não disponível")
        rows = build_reference_rows(path)
        self.assertEqual(len(rows), 75 * 60)
        self.assertTrue(all(row["answer"].count("; DOI:") == 1 for row in rows))


if __name__ == "__main__":
    unittest.main()
