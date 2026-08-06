import tempfile
import unittest
from pathlib import Path

import numpy as np

from security.controls import APIKeyAuth, RateLimiter, decrypt_file, encrypt_file, validate_input


class SecurityTest(unittest.TestCase):
    def test_auth_roles_and_rate_limit(self):
        auth = APIKeyAuth({"secret": "user", "admin-secret": "admin"})
        self.assertTrue(auth.authorize("secret"))
        self.assertFalse(auth.authorize("secret", "admin"))
        self.assertTrue(auth.authorize("admin-secret", "admin"))
        limiter = RateLimiter(limit=2, window_seconds=60)
        self.assertTrue(limiter.allow("client", now=0))
        self.assertTrue(limiter.allow("client", now=1))
        self.assertFalse(limiter.allow("client", now=2))

    def test_encryption_and_input_validation(self):
        from cryptography.fernet import Fernet

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "model.bin"
            encrypted = Path(temporary) / "model.bin.enc"
            restored = Path(temporary) / "model.restored"
            source.write_bytes(b"private model")
            key = Fernet.generate_key()
            encrypt_file(source, encrypted, key)
            decrypt_file(encrypted, restored, key)
            self.assertEqual(restored.read_bytes(), source.read_bytes())
        self.assertTrue(np.array_equal(validate_input(np.zeros((1, 2))), [[0, 0]]))
        with self.assertRaises(ValueError):
            validate_input(np.array([[np.nan]]))


if __name__ == "__main__":
    unittest.main()
