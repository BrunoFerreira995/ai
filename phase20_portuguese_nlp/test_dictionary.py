import unittest

from phase20_portuguese_nlp.dictionary import load_portuguese_words


class DictionaryTests(unittest.TestCase):
    def test_downloaded_dictionary_loads(self):
        words = load_portuguese_words()
        self.assertGreater(len(words), 100_000)
        self.assertIn("matemática", words)


if __name__ == "__main__":
    unittest.main()
