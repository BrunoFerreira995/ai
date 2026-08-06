import unittest

import numpy as np
import tensorflow as tf

from phase22_language_model import CharacterTokenizer, CausalLMConfig, DecoderOnlyCausalLM, InferenceEngine, beam_search, generate
from phase22_language_model.alignment import dpo_loss
from phase22_language_model.flash_attention import flash_attention_available
from phase22_language_model.vocabulary import make_causal_examples


class Phase22Tests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer().fit(["Olá mundo", "Matemática"])
        self.config = CausalLMConfig(vocab_size=self.tokenizer.vocab_size, max_sequence_length=32, hidden_size=16, num_layers=2, num_heads=4, num_kv_heads=1, intermediate_size=32)
        self.model = DecoderOnlyCausalLM(self.config)
        self.model(tf.constant([[self.tokenizer.bos_id, self.tokenizer.eos_id]]))

    def test_tokenizer_and_causal_examples(self):
        ids = self.tokenizer.encode("oi")
        self.assertEqual(self.tokenizer.decode(ids), "oi")
        x, y = make_causal_examples(self.tokenizer, ["oi"], 8)
        self.assertEqual(x.shape, y.shape)

    def test_decoder_and_generation(self):
        prompt = [[self.tokenizer.bos_id, self.tokenizer.vocabulary.index("O")]]
        result = self.model(prompt)
        self.assertEqual(result.shape[-1], self.tokenizer.vocab_size)
        self.assertEqual(generate(self.model, prompt, max_new_tokens=2).shape[1], 4)
        self.assertEqual(beam_search(self.model, prompt, num_beams=2, max_new_tokens=2).shape[1], 4)

    def test_engine_and_alignment(self):
        text = InferenceEngine(self.model, self.tokenizer).generate("O", max_new_tokens=1)
        self.assertIsInstance(text, str)
        self.assertGreaterEqual(float(dpo_loss(1.0, 0.0, 0.5, 0.1)), 0.0)
        self.assertIsInstance(flash_attention_available(), bool)


if __name__ == "__main__":
    unittest.main()
