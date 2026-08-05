"""Quantization, pruning, distillation, and weight clustering with TF-MOT."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot


def quantize_model(model: keras.Model) -> keras.Model:
    """Return a quantization-aware-training version of a Keras model."""
    return tfmot.quantization.keras.quantize_model(model)


def convert_to_int8_tflite(
    model: keras.Model,
    representative_dataset: Callable[[], Iterable[np.ndarray]],
) -> bytes:
    """Convert a model to fully integer TFLite using representative samples."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def prune_model(
    model: keras.Model,
    *,
    end_step: int,
    target_sparsity: float = 0.5,
    begin_step: int = 0,
) -> keras.Model:
    """Wrap eligible layers with a polynomial-magnitude pruning schedule."""
    if not 0 <= target_sparsity < 1:
        raise ValueError("target_sparsity must be in [0, 1)")
    if end_step <= begin_step:
        raise ValueError("end_step must be greater than begin_step")
    schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=target_sparsity,
        begin_step=begin_step,
        end_step=end_step,
    )
    return tfmot.sparsity.keras.prune_low_magnitude(model, pruning_schedule=schedule)


def strip_pruning(model: keras.Model) -> keras.Model:
    """Remove pruning wrappers before exporting a final model."""
    return tfmot.sparsity.keras.strip_pruning(model)


def cluster_model(model: keras.Model, clusters: int = 8) -> keras.Model:
    """Wrap eligible layers with weight clustering."""
    if clusters < 2:
        raise ValueError("clusters must be at least 2")
    return tfmot.clustering.keras.cluster_weights(
        model,
        number_of_clusters=clusters,
        cluster_centroids_init=tfmot.clustering.keras.CentroidInitialization.LINEAR,
    )


class Distiller(keras.Model):
    """Knowledge-distillation model combining hard and teacher soft targets."""

    def __init__(self, student: keras.Model, teacher: keras.Model, **kwargs):
        super().__init__(**kwargs)
        self.student = student
        self.teacher = teacher
        self.student_loss_fn = None
        self.distillation_loss_fn = keras.losses.KLDivergence()
        self.alpha = 0.1
        self.temperature = 3.0

    def compile(
        self,
        optimizer: keras.optimizers.Optimizer,
        student_loss_fn: Callable,
        *,
        alpha: float = 0.1,
        temperature: float = 3.0,
        metrics=None,
    ):
        if not 0 <= alpha <= 1 or temperature <= 0:
            raise ValueError("alpha must be in [0, 1] and temperature must be positive")
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.alpha = alpha
        self.temperature = temperature
        self.teacher.trainable = False

    def train_step(self, data):
        inputs, labels = data
        teacher_predictions = self.teacher(inputs, training=False)
        with tf.GradientTape() as tape:
            student_predictions = self.student(inputs, training=True)
            student_loss = self.student_loss_fn(labels, student_predictions)
            teacher_soft = tf.nn.softmax(teacher_predictions / self.temperature, axis=-1)
            student_soft = tf.nn.softmax(student_predictions / self.temperature, axis=-1)
            distillation_loss = self.distillation_loss_fn(teacher_soft, student_soft) * self.temperature**2
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        gradients = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))
        self.compiled_metrics.update_state(labels, student_predictions)
        return {"loss": loss, **{metric.name: metric.result() for metric in self.metrics}}

    def call(self, inputs, training=False):
        return self.student(inputs, training=training)
