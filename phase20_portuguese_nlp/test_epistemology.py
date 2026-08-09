import unittest

from .epistemology import epistemology_rows


class EpistemologyTests(unittest.TestCase):
    def test_subject_and_object_are_present(self):
        rows = epistemology_rows()
        text = " ".join(row["question"] + row["answer"] for row in rows).lower()
        self.assertIn("sujeito", text)
        self.assertIn("objeto", text)
        self.assertIn("justifica", text)


if __name__ == "__main__":
    unittest.main()
