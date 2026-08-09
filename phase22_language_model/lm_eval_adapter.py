"""lm-evaluation-harness adapter for the local TensorFlow causal LM."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from .generation import generate
from .model import CausalLMConfig, DecoderOnlyCausalLM
from .tokenizer import load_tokenizer

try:
    from lm_eval.api.model import LM
except ImportError:  # pragma: no cover - optional dependency
    class LM:  # type: ignore[no-redef]
        pass


class TensorFlowCausalLM(LM):
    """Minimal LM interface implemented by the project's own checkpoint."""

    def __init__(self, pretrained: str, device: str = "cpu", batch_size: int = 1, **kwargs):
        super().__init__()
        self.model_dir = Path(pretrained)
        self.device_name = device
        self._device = device
        self.batch_size = int(batch_size)
        self.tokenizer = load_tokenizer(self.model_dir / "tokenizer.json")
        config = CausalLMConfig(**json.loads((self.model_dir / "config.json").read_text(encoding="utf-8")))
        with tf.device(device):
            self.model = DecoderOnlyCausalLM(config)
            self.model(tf.zeros((1, 1), dtype=tf.int32))
            self.model.load_weights(self.model_dir / "model.weights.h5")

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_id

    @property
    def max_length(self):
        return self.model.config.max_sequence_length

    def tok_encode(self, string: str, add_special_tokens=True):
        return self.tokenizer.encode(string, add_special_tokens=add_special_tokens)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(list(tokens))

    def loglikelihood(self, requests):
        outputs = []
        for request in requests:
            context, continuation = request.args
            context_ids = self.tok_encode(context, add_special_tokens=True)
            if context_ids and context_ids[-1] == self.tokenizer.eos_id:
                context_ids = context_ids[:-1]
            continuation_ids = self.tok_encode(continuation, add_special_tokens=False)
            full_ids = context_ids + continuation_ids
            inputs = tf.constant([full_ids[:-1]], dtype=tf.int32)
            logits = self.model(inputs, training=False)[0]
            start = max(len(context_ids) - 1, 0)
            target = tf.constant(continuation_ids, dtype=tf.int32)
            selected = logits[start : start + len(continuation_ids)]
            log_probs = tf.nn.log_softmax(selected, axis=-1)
            positions = tf.stack([tf.range(len(continuation_ids)), target], axis=1)
            score = tf.reduce_sum(tf.gather_nd(log_probs, positions))
            outputs.append((float(score), True))
        return outputs

    def loglikelihood_rolling(self, requests):
        return [self.loglikelihood([(request)])[0][0] for request in requests]

    def generate_until(self, requests):
        outputs = []
        for request in requests:
            context, generation_kwargs = request.args
            until = generation_kwargs.get("until", [])
            max_tokens = int(generation_kwargs.get("max_gen_toks", 64))
            prompt = tf.constant([self.tok_encode(context, add_special_tokens=True)], dtype=tf.int32)
            generated = generate(self.model, prompt, max_new_tokens=max_tokens, eos_token_id=self.eot_token_id)
            text = self.tok_decode(generated[0].numpy().tolist())
            for stop in until:
                text = text.split(stop)[0]
            outputs.append(text)
        return outputs
