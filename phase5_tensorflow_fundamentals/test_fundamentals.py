import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from phase5_tensorflow_fundamentals.fundamentals import (
    build_dataset,
    graph_add,
    load_file_dataset,
    tensor_operations,
    variable_update,
)


class TensorFlowFundamentalsTest(unittest.TestCase):
    def test_tensors_variables_and_operations(self):
        operations = tensor_operations([1, 2, 3])
        self.assertTrue(np.array_equal(operations["squared"].numpy(), [1, 4, 9]))
        self.assertEqual(float(operations["sum"].numpy()), 6.0)
        self.assertEqual(float(variable_update(2, 3).numpy()), 5.0)

    def test_eager_and_graph_execution_with_broadcasting(self):
        result = graph_add(tf.constant([[1], [2]]), tf.constant([10, 20]))
        self.assertTrue(np.array_equal(result.numpy(), [[11, 21], [12, 22]]))

    def test_dataset_pipeline(self):
        dataset = build_dataset(range(10), labels=range(10), batch_size=4, cache=True, prefetch=True)
        batches = list(dataset.as_numpy_iterator())
        self.assertEqual(len(batches), 3)
        self.assertEqual(batches[0][0].tolist(), [0, 1, 2, 3])
        self.assertEqual(batches[-1][0].tolist(), [8, 9])

    def test_parallel_file_loader(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for value in ("a", "bb"):
                path = Path(temporary) / f"{value}.txt"
                path.write_text(value, encoding="utf-8")
                paths.append(str(path))

            def load(path):
                return tf.io.read_file(path)

            output = list(load_file_dataset(paths, load, batch_size=2).as_numpy_iterator())
            self.assertEqual(len(output), 1)
            self.assertEqual(set(output[0].tolist()), {b"a", b"bb"})


if __name__ == "__main__":
    unittest.main()
