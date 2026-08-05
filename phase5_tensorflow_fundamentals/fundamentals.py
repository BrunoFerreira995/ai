"""Small, executable examples of TensorFlow fundamentals."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import tensorflow as tf


def tensor_operations(values: Iterable[float]) -> dict[str, tf.Tensor]:
    """Create a tensor and return common tensor operations on it."""
    tensor = tf.convert_to_tensor(values, dtype=tf.float32)
    return {
        "tensor": tensor,
        "squared": tf.square(tensor),
        "sum": tf.reduce_sum(tensor),
        "mean": tf.reduce_mean(tensor),
        "shape": tf.shape(tensor),
    }


def variable_update(initial_value: float, delta: float) -> tf.Variable:
    """Create a trainable variable and update it in place."""
    variable = tf.Variable(initial_value, dtype=tf.float32, trainable=True)
    variable.assign_add(delta)
    return variable


@tf.function
def graph_add(left: tf.Tensor, right: tf.Tensor) -> tf.Tensor:
    """Add tensors through a traced graph function."""
    return tf.add(left, right)


def build_dataset(
    values: Iterable[object],
    labels: Iterable[object] | None = None,
    *,
    batch_size: int = 32,
    shuffle: bool = False,
    shuffle_buffer: int = 1000,
    cache: bool = False,
    prefetch: bool = True,
    map_function: Callable | None = None,
) -> tf.data.Dataset:
    """Build a configurable ``tf.data.Dataset`` pipeline.

    The order is map, optional cache, optional shuffle, batch, then prefetch.
    ``map_function`` receives either one value or ``(value, label)``.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    dataset = tf.data.Dataset.from_tensor_slices((values, labels) if labels is not None else values)
    if map_function is not None:
        dataset = dataset.map(map_function, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(buffer_size=max(1, shuffle_buffer), reshuffle_each_iteration=False)
    dataset = dataset.batch(batch_size)
    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def load_file_dataset(
    paths: Iterable[str],
    loader: Callable[[tf.Tensor], tf.Tensor],
    *,
    batch_size: int = 32,
    shuffle: bool = False,
    cache: bool = False,
) -> tf.data.Dataset:
    """Create a parallel file-loading pipeline from file paths.

    ``loader`` should read and decode one path. It is mapped with
    ``AUTOTUNE`` so TensorFlow can parallelize I/O and preprocessing.
    """
    path_list = list(paths)
    dataset = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(path_list, dtype=tf.string))
    dataset = dataset.map(loader, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(max(1, len(path_list)), reshuffle_each_iteration=False)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
