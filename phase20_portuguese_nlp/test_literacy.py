import unittest

from phase20_portuguese_nlp.literacy import analyze_word, split_syllables


class LiteracyTests(unittest.TestCase):
    def test_letters_and_word_classification(self):
        result = analyze_word("casa")
        self.assertEqual(result["letters"], ["c", "a", "s", "a"])
        self.assertEqual(result["syllables"], ["ca", "sa"])
        self.assertEqual(result["classification"], "dissílaba")

    def test_syllables(self):
        self.assertEqual(split_syllables("computador"), ["com", "pu", "ta", "dor"])


if __name__ == "__main__":
    unittest.main()
