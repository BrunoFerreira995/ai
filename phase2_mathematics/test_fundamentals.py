import unittest

import numpy as np

from phase2_mathematics.fundamentals import (
    bernoulli_pmf,
    bayes_theorem,
    categorical_cross_entropy,
    chain_rule,
    eigendecomposition,
    finite_difference_derivative,
    gaussian_pdf,
    gradient,
    hessian,
    jacobian,
    matrix_multiply,
    softmax,
)


class MathematicsTest(unittest.TestCase):
    def test_linear_algebra(self):
        self.assertTrue(np.array_equal(matrix_multiply([[1, 2]], [[3], [4]]), [[11]]))
        values, vectors = eigendecomposition(np.diag([2.0, 3.0]))
        self.assertTrue(np.allclose(np.sort(values), [2, 3]))
        self.assertEqual(vectors.shape, (2, 2))

    def test_calculus(self):
        self.assertAlmostEqual(finite_difference_derivative(lambda x: x**2, 3), 6, places=4)
        function = lambda point: point[0] ** 2 + 3 * point[1]
        self.assertTrue(np.allclose(gradient(function, [2, 4]), [4, 3], atol=1e-4))
        self.assertAlmostEqual(chain_rule(lambda x: 2 * x, lambda _: 3, 4), 24)
        self.assertTrue(np.allclose(jacobian(lambda point: [point[0] ** 2, point[1] + 1], [2, 3]), [[4, 0], [0, 1]], atol=1e-4))
        self.assertTrue(np.allclose(hessian(lambda point: point[0] ** 2 + 3 * point[1] ** 2, [2, 1]), [[2, 0], [0, 6]], atol=1e-3))

    def test_probability(self):
        self.assertAlmostEqual(bayes_theorem(0.2, 0.8, 0.4), 0.4)
        self.assertAlmostEqual(gaussian_pdf(0), 1 / np.sqrt(2 * np.pi))
        self.assertAlmostEqual(bernoulli_pmf(1, 0.7), 0.7)
        probabilities = softmax([1, 2, 3])
        self.assertAlmostEqual(float(np.sum(probabilities)), 1.0)
        self.assertAlmostEqual(categorical_cross_entropy([0, 1], [0.1, 0.9]), -np.log(0.9))


if __name__ == "__main__":
    unittest.main()
