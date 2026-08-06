"""Optional alignment losses (DPO and a compact RLHF reward objective)."""

from __future__ import annotations

import tensorflow as tf


def dpo_loss(policy_chosen_logp, policy_rejected_logp, reference_chosen_logp, reference_rejected_logp, beta: float = 0.1):
    """Direct Preference Optimization loss for chosen/rejected completions."""
    policy_margin = policy_chosen_logp - policy_rejected_logp
    reference_margin = reference_chosen_logp - reference_rejected_logp
    return -tf.reduce_mean(tf.math.log_sigmoid(beta * (policy_margin - reference_margin)))


def rlhf_reward_loss(rewards, values):
    """Simple squared reward-model objective used before policy optimization."""
    return tf.reduce_mean(tf.square(tf.cast(rewards, tf.float32) - tf.cast(values, tf.float32)))
