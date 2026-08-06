import unittest

import numpy as np

from phase20_advanced_ai.nlp import DictionaryTranslator, PortugueseTextClassifier
from phase20_advanced_ai.vision import FaceRecognizer, ImageClassifier, segment_instances


class FakeModel:
    def predict(self, values, verbose=0):
        return np.array([[0.1, 0.9]])


class AdvancedAITest(unittest.TestCase):
    def test_image_classification_and_segmentation(self):
        result = ImageClassifier(FakeModel(), ["cat", "dog"]).predict(np.zeros((8, 8, 3)))
        self.assertEqual(result["class_name"], "dog")
        masks = segment_instances(np.zeros((8, 8, 3)), lambda batch: np.zeros((1, 2, 8, 8)))
        self.assertEqual(masks.shape, (1, 2, 8, 8))

    def test_face_similarity_and_nlp(self):
        recognizer = FaceRecognizer(lambda image: np.array([1.0, 0.0]))
        self.assertTrue(recognizer.is_same_person(np.zeros(1), np.ones(1)))
        classifier = PortugueseTextClassifier().fit(["olá", "quero estudar matemática"], ["saudacao", "duvida_matematica"])
        self.assertEqual(classifier.predict("quero estudar matemática")[0], "duvida_matematica")
        self.assertEqual(DictionaryTranslator().translate("olá matemática"), "hello mathematics")


if __name__ == "__main__":
    unittest.main()
