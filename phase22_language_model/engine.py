"""Inference engine around the tokenizer and decoder-only model."""

from __future__ import annotations

from .generation import beam_search, generate


class InferenceEngine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt: str, *, max_new_tokens: int = 64, beam_size: int = 1, temperature: float = 0.0) -> str:
        encoded = [self.tokenizer.encode(prompt, add_special_tokens=True)]
        if beam_size > 1:
            output = beam_search(self.model, encoded, num_beams=beam_size, max_new_tokens=max_new_tokens, eos_token_id=self.tokenizer.eos_id)
        else:
            output = generate(self.model, encoded, max_new_tokens=max_new_tokens, eos_token_id=self.tokenizer.eos_id, temperature=temperature)
        return self.tokenizer.decode(output[0].numpy().tolist())
