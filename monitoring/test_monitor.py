import unittest

from monitoring.monitor import AlertManager, DriftDetector, PerformanceMonitor, ResourceMonitor


class MonitoringTest(unittest.TestCase):
    def test_drift(self):
        detector = DriftDetector([0, 0, 0, 1, 1, 1])
        self.assertFalse(detector.has_drift([0, 0, 1, 1]))
        self.assertTrue(detector.has_drift([10, 10, 11, 11]))

    def test_performance_and_resources(self):
        monitor = PerformanceMonitor()
        self.assertEqual(monitor.measure(lambda: 2 + 2), 4)
        self.assertEqual(monitor.summary()["requests"], 1.0)
        self.assertIn("memory_percent", ResourceMonitor().snapshot())

    def test_alerts(self):
        received = []
        manager = AlertManager(handlers=[received.append])
        alerts = manager.evaluate(latency_ms=600, accuracy=0.5)
        self.assertEqual(len(alerts), 2)
        self.assertEqual(len(received), 2)


if __name__ == "__main__":
    unittest.main()
